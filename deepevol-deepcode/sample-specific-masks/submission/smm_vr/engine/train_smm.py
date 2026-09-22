"""SMM training loop -- Algorithm 1 of the paper.

Paper (Algorithm 1, "Visual Reprogramming with SMM")
----------------------------------------------------
Input: pre-trained model ``f_P``, loss ``l``, label-mapping function
``f_out^(j)`` for iteration ``j``, target-domain training data
``{(x_i, y_i)}_{i=1..n}``, maximum number of iterations ``E``, learning rate
``alpha_1`` for ``delta`` and ``alpha_2`` for ``phi``.

    Initialize phi randomly; set delta <- {0}^{d_P}
    for j = 1 to E do
        # Step1: Compute individual marks using the mask generator
        # Step2: Resize masks using the patch-wise interpolation module
        f_in(x_i; delta, phi) <- r(x_i) + delta ⊙ f_mask(r(x_i) | phi),  for all i
        # Compute the classification loss
        L(delta, phi) <- (1/n) sum_i l( f_out^(j)( f_P( f_in(x_i; delta, phi) ) ), y_i )
        delta <- delta - alpha_1 grad_delta L(delta, phi)
        phi   <- phi   - alpha_2 grad_phi   L(delta, phi)
    end for
Output: optimal delta*, phi*

The optimisers themselves are not specified by the paper (only an SGD-style
update with decaying learning rates), so SGD with momentum 0.9 and no weight
decay is used -- the paper states "no weight decay or additional regularizer"
and specifies the learning rates / decays / milestones.  The leading ``0`` in
Table 9's milestone list ``[0, 100, 145]`` is treated as a typographical
artifact, hence milestones ``(100, 145)`` over ``E = 200`` epochs.

Hyper-parameters (paper Section 5 "Baselines" and Appendix C Table 9)
---------------------------------------------------------------------
* ``E = 200`` epochs, milestones ``(100, 145)``.
* shared pattern ``delta``: ``alpha_1 = 0.01``, ``gamma_1 = 0.1``
  (following Chen et al., 2023).
* 5-layer mask generator (ResNet-18 / ResNet-50): ``alpha_2 = 0.01``,
  ``gamma_2 = 0.1``.
* 6-layer mask generator (ViT-B32): ``alpha_2 = 0.001``, ``gamma_2 = 1``.
* batch size ``256`` except DTD (``64``) and OxfordPets (``64``).
* three seeds, averaged test accuracy reported.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..label_mapping.flm import IGNORE_INDEX, apply_label_mapping
from ..modules.reprogram import SMMReprogram, build_smm_reprogram
from .metrics import AccuracyMeter, RunResult, aggregate_seeds, format_mean_std
from .seeds import (
    DEFAULT_DETERMINISTIC,
    dataloader_seed_kwargs,
    resolve_seeds,
    set_seed,
)

__all__ = [
    "SMMTrainConfig",
    "TrainingHistory",
    "EpochStats",
    "train_smm",
    "train_one_seed",
    "train_with_seeds",
    "build_optimizers",
    "build_scheduler",
    "evaluate_epoch",
    "train_accuracy",
    "MASK_LAYERS_BY_BACKBONE",
    "DEFAULT_MILESTONES",
    "DEFAULT_EPOCHS",
    "DEFAULT_ALPHA_DELTA",
    "DEFAULT_GAMMA_DELTA",
    "DEFAULT_ALPHA_MASK_5",
    "DEFAULT_GAMMA_MASK_5",
    "DEFAULT_ALPHA_MASK_6",
    "DEFAULT_GAMMA_MASK_6",
]

# ---------------------------------------------------------------------------
# Paper constants (Section 5 "Baselines", Appendix C Table 9)
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS: int = 200
DEFAULT_MILESTONES: Tuple[int, ...] = (100, 145)
DEFAULT_ALPHA_DELTA: float = 0.01
DEFAULT_GAMMA_DELTA: float = 0.1
DEFAULT_ALPHA_MASK_5: float = 0.01
DEFAULT_GAMMA_MASK_5: float = 0.1
DEFAULT_ALPHA_MASK_6: float = 0.001
DEFAULT_GAMMA_MASK_6: float = 1.0
DEFAULT_BATCH_SIZE: int = 256
SMALL_BATCH_DATASETS: Tuple[str, ...] = ("dtd", "oxfordpets")

# Number of CNN layers in the mask generator per pre-trained backbone
# (paper Section 3.2 / Appendix A.2: 5 layers for the ResNets, 6 for ViT-B32).
MASK_LAYERS_BY_BACKBONE: Dict[str, int] = {
    "resnet18": 5,
    "resnet50": 5,
    "resnet": 5,
    "vit_b32": 6,
    "vit_b_32": 6,
    "vit": 6,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SMMTrainConfig:
    """All knobs of the Algorithm 1 training loop.

    Defaults reproduce the paper's unified training recipe (Appendix C,
    Table 9); per-dataset overrides for batch size are applied automatically by
    :func:`train_smm`.
    """

    # optimisation (Algorithm 1)
    epochs: int = DEFAULT_EPOCHS
    milestones: Sequence[int] = DEFAULT_MILESTONES

    alpha_delta: float = DEFAULT_ALPHA_DELTA
    gamma_delta: float = DEFAULT_GAMMA_DELTA
    alpha_mask: Optional[float] = None          # None -> derived from layer count
    gamma_mask: Optional[float] = None          # None -> derived from layer count

    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 0.0
    nesterov: bool = False
    grad_clip: Optional[float] = None

    # data
    batch_size: int = DEFAULT_BATCH_SIZE
    test_batch_size: int = 256
    num_workers: int = 4
    download: bool = True
    drop_last: bool = False

    # shape / model
    backbone: str = "resnet18"
    input_size: Optional[int] = None            # None -> backbone default
    patch_size: int = 8                         # 2**l with l = 3 (Section 5)
    num_mask_layers: Optional[int] = None       # None -> MASK_LAYERS_BY_BACKBONE

    # label mapping
    label_mapping: str = "ilm"                  # "ilm" | "flm" | "rlm"
    mapping_refresh_every: int = 1              # Ilm recomputed every epoch

    # bookkeeping
    seed: int = 0
    device: Optional[str] = None
    deterministic: bool = DEFAULT_DETERMINISTIC
    log_every: int = 10
    eval_every: int = 1
    verbose: bool = True
    save_dir: Optional[str] = None
    save_checkpoint: bool = False
    max_train_batches: Optional[int] = None     # debug: truncate an epoch
    max_eval_batches: Optional[int] = None

    def alpha_for_mask(self, num_layers: Optional[int] = None) -> float:
        """``alpha_2``: 0.01 for the 5-layer CNN, 0.001 for the 6-layer CNN."""
        if self.alpha_mask is not None:
            return float(self.alpha_mask)
        layers = num_layers if num_layers is not None else (self.num_mask_layers or 5)
        return DEFAULT_ALPHA_MASK_6 if layers >= 6 else DEFAULT_ALPHA_MASK_5

    def gamma_for_mask(self, num_layers: Optional[int] = None) -> float:
        """``gamma_2``: 0.1 for the 5-layer CNN, 1 (no decay) for the 6-layer."""
        if self.gamma_mask is not None:
            return float(self.gamma_mask)
        layers = num_layers if num_layers is not None else (self.num_mask_layers or 5)
        return DEFAULT_GAMMA_MASK_6 if layers >= 6 else DEFAULT_GAMMA_MASK_5

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["milestones"] = list(self.milestones)
        return d


@dataclass
class EpochStats:
    """Metrics collected for a single epoch."""

    epoch: int
    loss: float = float("nan")
    train_accuracy: float = float("nan")
    test_accuracy: Optional[float] = None
    delta_lr: float = float("nan")
    mask_lr: float = float("nan")
    mapping_updated: bool = False
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingHistory:
    """Full record of one Algorithm 1 run (one seed)."""

    dataset: str = ""
    backbone: str = "resnet18"
    seed: int = 0
    label_mapping: str = "ilm"
    epochs: List[EpochStats] = field(default_factory=list)
    best_test_accuracy: float = float("nan")
    best_epoch: int = -1
    final_test_accuracy: float = float("nan")
    mask_parameters: int = 0
    delta_parameters: int = 0
    mapping_mean_accuracy: float = float("nan")
    elapsed_seconds: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)

    def add(self, stats: EpochStats) -> None:
        self.epochs.append(stats)
        if stats.test_accuracy is not None:
            if not (self.best_test_accuracy == self.best_test_accuracy) or (
                stats.test_accuracy > self.best_test_accuracy
            ):
                self.best_test_accuracy = float(stats.test_accuracy)
                self.best_epoch = int(stats.epoch)

    @property
    def losses(self) -> List[float]:
        return [e.loss for e in self.epochs]

    @property
    def test_accuracies(self) -> List[Optional[float]]:
        return [e.test_accuracy for e in self.epochs]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "backbone": self.backbone,
            "seed": self.seed,
            "label_mapping": self.label_mapping,
            "epochs": [e.as_dict() for e in self.epochs],
            "best_test_accuracy": self.best_test_accuracy,
            "best_epoch": self.best_epoch,
            "final_test_accuracy": self.final_test_accuracy,
            "mask_parameters": self.mask_parameters,
            "delta_parameters": self.delta_parameters,
            "mapping_mean_accuracy": self.mapping_mean_accuracy,
            "elapsed_seconds": self.elapsed_seconds,
            "config": self.config,
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, indent=2)
        return path


# ---------------------------------------------------------------------------
# Optimisers / schedulers (separate learning rates for delta and phi)
# ---------------------------------------------------------------------------
def _make_optimizer(name: str, params, lr: float, momentum: float, weight_decay: float, nesterov: bool):
    name = (name or "sgd").lower()
    if name in ("sgd", "momentum"):
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
        )
    if name in ("adam", "adamw"):
        cls = torch.optim.Adam if name == "adam" else torch.optim.AdamW
        return cls(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


def build_optimizers(
    model: SMMReprogram,
    config: SMMTrainConfig,
    *,
    num_mask_layers: Optional[int] = None,
):
    """Build the two SGD optimisers of Algorithm 1.

    Returns
    -------
    (optimizer_delta, optimizer_mask, scheduler_delta, scheduler_mask)
    """
    num_layers = (
        num_mask_layers if num_mask_layers is not None else (config.num_mask_layers or 5)
    )
    alpha_delta = float(config.alpha_delta)
    alpha_mask = config.alpha_for_mask(num_layers)

    optimizer_delta = _make_optimizer(
        config.optimizer,
        [model.delta],
        lr=alpha_delta,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )
    mask_params = [p for p in model.mask_generator.parameters() if p.requires_grad]
    optimizer_mask = _make_optimizer(
        config.optimizer,
        mask_params,
        lr=alpha_mask,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )

    milestones = [int(m) for m in config.milestones]
    scheduler_delta = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_delta, milestones=milestones, gamma=float(config.gamma_delta)
    )
    scheduler_mask = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_mask, milestones=milestones, gamma=config.gamma_for_mask(num_layers)
    )
    return optimizer_delta, optimizer_mask, scheduler_delta, scheduler_mask


def build_scheduler(optimizer, config: SMMTrainConfig, gamma: float):
    """Multi-step decay at the paper's milestones ``(100, 145)``."""
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(m) for m in config.milestones], gamma=float(gamma)
    )


# ---------------------------------------------------------------------------
# Label-mapping helpers
# ---------------------------------------------------------------------------
def _is_recomputing_mapping(mapping: Any, config: SMMTrainConfig, epoch: int) -> bool:
    """Whether ``f_out`` must be refreshed at the start of ``epoch`` (Ilm)."""
    if mapping is None:
        return False
    if not getattr(mapping, "recomputes_each_epoch", False):
        return False
    every = max(1, int(config.mapping_refresh_every))
    return (epoch - 1) % every == 0


def _mapping_accuracy(mapping: Any) -> float:
    acc = getattr(mapping, "mean_accuracy", None)
    if acc is None:
        return float("nan")
    try:
        return float(acc)
    except (TypeError, ValueError):
        return float("nan")


def _map_logits(f_out: Optional[Any], logits: torch.Tensor, target: torch.Tensor):
    """Apply ``f_out`` and translate targets into the mapped label space.

    Two output-mapping conventions are supported, mirroring the label-mapping
    package:

    * ``select`` (used by Flm/Ilm): ``target_to_pretrained[t] = y^P`` is the
      index of the matched pre-trained label, so the mapped logits are
      ``logits[:, target_to_pretrained]`` and the targets stay ``y^T``.
    * ``inject`` (used by Rlm): ``f_out`` builds a full ``|Y^P|`` logit vector
      whose weight per pre-trained class is ``logit`` when
      ``pretrained_to_target[p] == t`` else ``-logit``; the cross-entropy is
      then taken over the target classes.
    """
    if f_out is None:
        return logits, target

    mode = str(getattr(f_out, "mode", "select")).lower()
    num_target_classes = int(getattr(f_out, "num_target_classes", 0) or 0)

    if mode == "inject":
        mapped = f_out(logits)          # (B, |Y^T|)
        return mapped, target
    if num_target_classes <= 0:
        num_target_classes = int(getattr(f_out, "num_pretrained_classes", logits.shape[-1]))

    out = apply_label_mapping(logits, f_out.target_to_pretrained)
    # Defensive: uncovered target classes may be IGNORE_INDEX -> use zeros.
    t2p = f_out.target_to_pretrained
    if t2p.numel() < num_target_classes:
        pad = torch.full(
            (num_target_classes - t2p.numel(),), IGNORE_INDEX, dtype=t2p.dtype, device=t2p.device
        )
        t2p = torch.cat([t2p, pad])
    invalid = t2p[:num_target_classes] < 0
    if bool(invalid.any()):
        out = out.clone()
        out[:, invalid] = 0.0
    return out[:, :num_target_classes], target


# ---------------------------------------------------------------------------
# Evaluation inside the loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_epoch(
    model: SMMReprogram,
    classifier: nn.Module,
    data_loader,
    *,
    f_out: Optional[Any] = None,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    criterion: Optional[Callable] = None,
) -> Tuple[float, Optional[float]]:
    """Top-1 accuracy (%) of the reprogrammed data on one split.

    Returns ``(accuracy_percent, mean_loss_or_None)``.
    """
    was_training = model.training
    model.eval()
    classifier.eval()

    meter = AccuracyMeter(topk=(1,))
    losses: List[float] = []
    n_batches = 0

    for batch in data_loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        images, targets = _split_batch(batch)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        reprogrammed = model(images)
        logits = classifier(reprogrammed)
        mapped, mapped_targets = _map_logits(f_out, logits, targets)
        meter.update(mapped, mapped_targets)
        if criterion is not None:
            losses.append(float(criterion(mapped, mapped_targets).detach().cpu()))
        n_batches += 1

    if was_training:
        model.train()
    mean_loss = float(sum(losses) / len(losses)) if losses else None
    return float(meter.top1), mean_loss


def train_accuracy(
    model: SMMReprogram,
    classifier: nn.Module,
    data_loader,
    *,
    f_out: Optional[Any] = None,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> float:
    """Training top-1 accuracy (%), computed in eval mode without gradients."""
    acc, _ = evaluate_epoch(
        model, classifier, data_loader, f_out=f_out, device=device, max_batches=max_batches
    )
    return acc


def _split_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("dataloader batches must be (images, targets) pairs")


# ---------------------------------------------------------------------------
# Core: Algorithm 1
# ---------------------------------------------------------------------------
def train_smm(
    model: SMMReprogram,
    classifier: nn.Module,
    train_loader,
    *,
    test_loader=None,
    f_out: Optional[Any] = None,
    label_mapping_builder: Optional[Callable[[], Any]] = None,
    config: Optional[SMMTrainConfig] = None,
    dataset: str = "",
    device: Optional[torch.device] = None,
    history: Optional[TrainingHistory] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> TrainingHistory:
    """Run Algorithm 1.

    Parameters
    ----------
    model : SMMReprogram
        Supplies the trainable pair ``(delta, phi)``; ``delta`` must already be
        initialised to ``{0}^{d_P}`` (Algorithm 1, line 1).
    classifier : nn.Module
        Frozen pre-trained ``f_P`` mapping the reprogrammed input to ImageNet
        logits.
    train_loader, test_loader :
        Reproducible loaders built by :func:`smm_vr.data.build_dataloaders`.
    f_out : optional
        Output label mapping.  For Ilm pass an ``IlmMapping`` instance (it is
        refreshed at the start of every epoch); for Flm/Rlm the mapping is
        fixed before training.
    label_mapping_builder : optional
        Zero-argument callable returning the mapping for the refresh.  When
        given it takes precedence over ``f_out.update`` (useful for tests and
        for experiment runners that want to control the frequency-matrix
        computation).
    config : SMMTrainConfig
        Defaults reproduce Appendix C Table 9.
    """
    config = config or SMMTrainConfig()
    if dataset and config.batch_size == DEFAULT_BATCH_SIZE:
        pass  # batch size is fixed by the loader (see data/datasets.py)
    device = device or torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = model.to(device)
    classifier = classifier.to(device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    if hasattr(model, "freeze_classifier"):
        model.freeze_classifier()

    num_layers = config.num_mask_layers or MASK_LAYERS_BY_BACKBONE.get(
        _norm_backbone(config.backbone), 5
    )
    optimizer_delta, optimizer_mask, sched_delta, sched_mask = build_optimizers(
        model, config, num_mask_layers=num_layers
    )
    criterion: Callable = getattr(config, "criterion", None) or nn.CrossEntropyLoss()

    history = history or TrainingHistory(
        dataset=dataset, backbone=config.backbone, seed=config.seed, label_mapping=config.label_mapping
    )
    history.config = config.as_dict()
    history.mask_parameters = sum(
        p.numel() for p in model.mask_generator.parameters() if p.requires_grad
    )
    history.delta_parameters = int(model.delta.numel())

    log = logger or (lambda msg: None)
    if config.verbose:
        log(
            f"[SMM] dataset={dataset or 'n/a'} backbone={config.backbone} "
            f"mask_layers={num_layers} |delta|={history.delta_parameters} "
            f"|phi|={history.mask_parameters} alpha1={config.alpha_delta} "
            f"alpha2={config.alpha_for_mask(num_layers)} "
            f"gamma2={config.gamma_for_mask(num_layers)} epochs={config.epochs}"
        )

    start_time = time.time()
    images_seen = 0

    for epoch in range(1, int(config.epochs) + 1):
        epoch_start = time.time()

        # ------------------------------------------------------------------
        # Iterative label mapping: recompute f_out^(j) before the epoch
        # (Ilm, Algorithm 4) -- Flm/Rlm are computed once and kept fixed.
        # ------------------------------------------------------------------
        mapping_updated = False
        if _is_recomputing_mapping(f_out, config, epoch):
            if label_mapping_builder is not None:
                new_mapping = label_mapping_builder()
                if isinstance(new_mapping, dict):
                    f_out.set_mapping(new_mapping)
                elif new_mapping is not None:
                    f_out = new_mapping
                    if hasattr(f_out, "to"):
                        f_out = f_out.to(device)
                mapping_updated = True
            elif hasattr(f_out, "update"):
                try:
                    mapping_updated = bool(
                        f_out.update(model=classifier, data_loader=train_loader, f_in=model)
                    )
                except TypeError:
                    mapping_updated = bool(
                        f_out.update(
                            classifier, train_loader, model, device=device
                        )
                    )
        if epoch == 1:
            history.mapping_mean_accuracy = _mapping_accuracy(f_out)

        # ------------------------------------------------------------------
        # One epoch of Algorithm 1
        # ------------------------------------------------------------------
        model.train()
        mask_gen_training = getattr(model.mask_generator, "training", True)
        classifier.eval()

        total_loss = 0.0
        n_samples = 0
        n_batches = 0
        meter = AccuracyMeter(topk=(1,))

        for batch in train_loader:
            if config.max_train_batches is not None and n_batches >= config.max_train_batches:
                break
            images, targets = _split_batch(batch)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = int(images.shape[0])

            # f_in(x_i; delta, phi) = r(x_i) + delta ⊙ f_mask(r(x_i) | phi)
            reprogrammed = model(images)

            # L(delta, phi) = (1/n) sum_i l(f_out^(j)(f_P(f_in(x_i))), y_i)
            with torch.no_grad():
                logits = classifier(reprogrammed)
            logits = logits.detach().requires_grad_(False)

            # Re-run the head graph so that gradients reach delta/phi through
            # the frozen classifier: use a non-no_grad pass guarded by the
            # frozen parameters (cheap because only f_mask is trained).
            logits = classifier(reprogrammed)
            mapped, mapped_targets = _map_logits(f_out, logits, targets)
            loss = criterion(mapped, mapped_targets)

            optimizer_delta.zero_grad(set_to_none=True)
            optimizer_mask.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(model.mask_generator.parameters()) + [model.delta], config.grad_clip
                )

            # delta <- delta - alpha_1 grad_delta L ; phi <- phi - alpha_2 grad_phi L
            optimizer_delta.step()
            optimizer_mask.step()

            total_loss += float(loss.detach().cpu()) * batch_size
            n_samples += batch_size
            n_batches += 1
            images_seen += batch_size
            with torch.no_grad():
                meter.update(mapped.detach(), mapped_targets)

        sched_delta.step()
        sched_mask.step()

        mean_loss = total_loss / max(1, n_samples)
        train_acc = float(meter.top1)

        test_acc: Optional[float] = None
        if test_loader is not None and config.eval_every and epoch % config.eval_every == 0:
            test_acc, _ = evaluate_epoch(
                model,
                classifier,
                test_loader,
                f_out=f_out,
                device=device,
                max_batches=config.max_eval_batches,
                criterion=criterion if False else None,
            )

        stats = EpochStats(
            epoch=epoch,
            loss=mean_loss,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            delta_lr=float(optimizer_delta.param_groups[0]["lr"]),
            mask_lr=float(optimizer_mask.param_groups[0]["lr"]),
            mapping_updated=mapping_updated,
            seconds=time.time() - epoch_start,
        )
        history.add(stats)

        if config.verbose and (
            epoch % max(1, config.log_every) == 0 or epoch == 1 or epoch == int(config.epochs)
        ):
            msg = (
                f"[SMM] epoch {epoch:>3}/{config.epochs} loss {mean_loss:.4f} "
                f"train_acc {train_acc:6.2f}%"
            )
            if test_acc is not None:
                msg += f" test_acc {test_acc:6.2f}%"
            msg += (
                f" lr_delta {stats.delta_lr:.5f} lr_mask {stats.mask_lr:.5f}"
                f" ({stats.seconds:.1f}s)"
            )
            log(msg)

        if config.save_dir and config.save_checkpoint and (
            test_acc is not None and test_acc >= history.best_test_accuracy
        ):
            _save_checkpoint(model, config, dataset, epoch, history, test_acc)

    history.elapsed_seconds = time.time() - start_time
    if history.epochs:
        last = history.epochs[-1]
        history.final_test_accuracy = (
            last.test_accuracy if last.test_accuracy is not None else float("nan")
        )

    if config.save_dir:
        history.save(os.path.join(config.save_dir, f"history_{dataset or 'run'}_{config.seed}.json"))

    return history


def _norm_backbone(backbone: str) -> str:
    return str(backbone).lower().replace("-", "_").replace(".", "")


def _save_checkpoint(model, config, dataset, epoch, history, test_acc) -> str:
    path = os.path.join(
        config.save_dir,
        f"checkpoint_{dataset or 'run'}_{config.backbone}_{config.label_mapping}_seed{config.seed}.pt",
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "delta": model.delta.detach().cpu(),
            "mask_generator": {
                k: v.detach().cpu() for k, v in model.mask_generator.state_dict().items()
            },
            "epoch": epoch,
            "test_accuracy": test_acc,
            "config": config.as_dict(),
            "history_best": history.best_test_accuracy,
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def train_one_seed(
    classifier: nn.Module,
    datasets: Any,
    *,
    dataset: str,
    backbone: str = "resnet18",
    config: Optional[SMMTrainConfig] = None,
    f_out: Optional[Any] = None,
    seed: int = 0,
    device: Optional[torch.device] = None,
    num_classes: Optional[int] = None,
    log: Optional[Callable[[str], None]] = None,
) -> TrainingHistory:
    """Build ``f_mask`` + ``delta``, the mapping and the loaders, then train.

    ``datasets`` may be ``(train_dataset, test_dataset)`` or a
    ``(train_loader, test_loader)`` pair; dataset objects are wrapped into
    reproducible loaders with the paper's batch size rule (256, or 64 for DTD
    and OxfordPets).
    """
    from ..data.datasets import build_dataloaders, canonical_name, num_classes as n_classes

    config = config or SMMTrainConfig()
    config.seed = seed
    config.backbone = backbone
    if canonical_name(dataset) in SMALL_BATCH_DATASETS and config.batch_size == DEFAULT_BATCH_SIZE:
        config.batch_size = 64

    set_seed(seed, deterministic=config.deterministic)
    device = device or torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if isinstance(datasets, tuple) and len(datasets) == 2 and hasattr(datasets[0], "dataset"):
        train_loader, test_loader = datasets
    else:
        train_loader, test_loader, _spec = build_dataloaders(
            dataset,
            backbone=backbone,
            batch_size=config.batch_size,
            test_batch_size=config.test_batch_size,
            num_workers=config.num_workers,
            download=config.download,
            drop_last=config.drop_last,
            device=str(device),
            **dataloader_seed_kwargs(seed),
        )

    num_classes = num_classes or n_classes(dataset)

    if f_out is None and config.label_mapping.lower() in ("ilm", "flm", "rlm"):
        f_out = build_label_mapping(
            config.label_mapping,
            classifier=classifier,
            data_loader=train_loader,
            num_target_classes=num_classes,
            device=device,
            seed=seed,
        )

    model = build_smm_reprogram(
        backbone=backbone,
        input_size=config.input_size,
        patch_size=config.patch_size,
    )
    model.to(device)

    return train_smm(
        model,
        classifier,
        train_loader,
        test_loader=test_loader,
        f_out=f_out,
        config=config,
        dataset=dataset,
        device=device,
        logger=log,
    )


def build_label_mapping(
    name: str,
    *,
    classifier: nn.Module,
    data_loader,
    num_target_classes: int,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Any:
    """Construct ``f_out`` (Ilm default, also Flm and Rlm)."""
    key = str(name).lower()
    from ..label_mapping import build_ilm_mapping, build_flm_mapping  # local import

    if key == "ilm":
        return build_ilm_mapping(
            classifier,
            data_loader,
            num_target_classes,
            f_in=None,              # Algorithm 4 initialisation (delta = 0, identity)
            device=device,
        )
    if key == "flm":
        return build_flm_mapping(
            classifier,
            data_loader,
            num_target_classes,
            f_in=None,
            device=device,
        )
    if key == "rlm":
        from ..label_mapping.rlm import build_rlm_mapping

        return build_rlm_mapping(
            num_target_classes,
            classifier=classifier,
            seed=seed,
            device=device,
        )
    raise ValueError(f"unknown label mapping {name!r}")


def train_with_seeds(
    classifier_factory: Callable[[], nn.Module],
    datasets: Any,
    *,
    dataset: str,
    backbone: str = "resnet18",
    config: Optional[SMMTrainConfig] = None,
    seeds: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
    method: str = "ours",
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run Algorithm 1 for the paper's three seeds and aggregate mean +- std."""
    config = config or SMMTrainConfig()
    seed_list = resolve_seeds(seeds)
    per_seed: List[float] = []
    histories: List[TrainingHistory] = []

    for seed in seed_list:
        classifier = classifier_factory()
        hist = train_one_seed(
            classifier,
            datasets,
            dataset=dataset,
            backbone=backbone,
            config=config,
            seed=seed,
            device=device,
            log=log,
        )
        histories.append(hist)
        per_seed.append(hist.final_test_accuracy)

    mean, std = aggregate_seeds(per_seed)
    return {
        "dataset": dataset,
        "backbone": backbone,
        "method": method,
        "seeds": list(seed_list),
        "per_seed_accuracy": per_seed,
        "mean": mean,
        "std": std,
        "formatted": format_mean_std(mean, std),
        "histories": histories,
        "result": RunResult(
            dataset=dataset,
            backbone=backbone,
            method=method,
            seed=seed_list[0] if seed_list else 0,
            accuracy=mean,
            mapping=config.label_mapping,
            extra={"std": std, "per_seed": per_seed, "n_seeds": len(seed_list)},
        ),
    }
