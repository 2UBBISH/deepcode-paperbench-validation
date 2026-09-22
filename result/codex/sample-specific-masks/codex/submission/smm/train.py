"""Training loop of SMM (Algorithm 1) and of the shared-mask baselines.

Paper reference: Section 3.4 (learning strategy), Section 5 + Appendix C
(optimisation settings: 200 epochs, initial learning rate 0.01, decay 0.1 at
epochs 100 and 145 for the ResNets; 0.001 with no decay for ViT-B/32).

Two parameter groups are optimised jointly:

* ``delta``, the shared noise pattern (learning rate ``lr_delta``), and
* ``phi``, the parameters of the mask generator ``f_mask`` (learning rate
  ``lr_mask``, Table 9) -- empty for the shared-mask baselines.

The label mapping ``f_out`` is refreshed every epoch (Ilm), every
``ilm_every_n_epochs`` epochs, or fixed at initialisation (Flm/Rlm).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .datasets import DATASET_SPECS, build_dataset
from .label_mapping import (
    LabelMapping,
    frequent_label_mapping,
    greedy_injective_mapping,
    frequency_distribution,
    random_label_mapping,
)
from .models import build_backbone
from .reprogram import build_input_reprogramming

__all__ = ["TrainConfig", "ReprogramModel", "train" ]


@dataclass
class TrainConfig:
    """Everything needed to reproduce one cell of Table 1/2/3/10."""

    dataset: str = "cifar10"
    backbone: str = "resnet18"
    image_size: int = 224
    method: str = "smm"                 # smm | full | narrow | medium | pad | only_mask | single_channel
    label_mapping: str = "ilm"          # ilm | flm | rlm
    num_layers: int = 5                 # mask generator depth (5 ResNet, 6 ViT)
    num_pool_layers: int = 3            # patch size 2 ** l (Figure 4)
    epochs: int = 200
    batch_size: Optional[int] = None      # None -> Table 9 batch size of the dataset
    lr_delta: float = 0.01
    lr_mask: float = 0.01
    gamma_delta: float = 0.1
    gamma_mask: float = 0.1
    milestones: Tuple[int, ...] = (100, 145)
    momentum: float = 0.9
    weight_decay: float = 0.0
    seed: int = 0
    num_workers: int = 4
    data_root: str = "data"
    download: bool = True
    split_dir: Optional[str] = None
    device: str = "cuda"
    output_dir: str = "runs"
    eval_every: int = 10
    log_every: int = 50
    ilm_every_n_epochs: int = 1
    pretrained: bool = True
    arch: str = "torchvision"           # ViT: "torchvision" (224 ckpt, resized pos-emb) or "timm_384"
    pad_width: int = 28                 # border width of the Pad baseline (Section 5)
    border_width: int = 28              # Narrow mask width
    medium_width: int = 56              # Medium mask width
    hidden_channels: Optional[Sequence[int]] = None  # mask generator widths (Figures 8/9)
    train_subset: Optional[int] = None  # use only N training samples (fast sanity runs)
    test_subset: Optional[int] = None   # use only N testing samples
    save_masks_every: int = 0           # >0 stores mask statistics for analysis
    run_name: Optional[str] = None

    def resolved_image_size(self) -> int:
        if self.image_size:
            return int(self.image_size)
        return 384 if "vit" in self.backbone else 224

    def resolved_batch_size(self) -> int:
        if self.batch_size:
            return int(self.batch_size)
        return DATASET_SPECS[self.dataset].batch_size


class ReprogramModel(nn.Module):
    """Frozen pre-trained model + trainable input transformation ``f_in``."""

    def __init__(
        self,
        backbone_name: str,
        dataset: str,
        method: str = "smm",
        image_size: Optional[int] = None,
        num_layers: int = 5,
        num_pool_layers: int = 3,
        pretrained: bool = True,
        device: str = "cpu",
        arch: str = "torchvision",
        pad_width: int = 28,
        border_width: int = 28,
        medium_width: int = 56,
        hidden_channels: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.bundle = build_backbone(
            backbone_name, pretrained=pretrained, image_size=image_size, arch=arch
        )
        self.image_size = self.bundle.image_size
        self.dataset = dataset
        self.num_target_classes = DATASET_SPECS[dataset].num_classes
        self.input_transform = build_input_reprogramming(
            image_size=(self.image_size, self.image_size),
            method=method,
            num_layers=num_layers,
            num_pool_layers=num_pool_layers,
            pad_width=pad_width,
            border_width=border_width,
            medium_width=medium_width,
            hidden_channels=hidden_channels,
        )
        self.method = method

    # ------------------------------------------------------------ properties
    @property
    def backbone(self) -> nn.Module:
        return self.bundle.model

    @property
    def num_pretrained_classes(self) -> int:
        return self.bundle.num_pretrained_classes

    @property
    def delta(self) -> Optional[nn.Parameter]:
        return self.input_transform.delta

    def mask_parameters(self):
        return self.input_transform.mask_parameters()

    def trainable_parameters(self):
        params = list(self.mask_parameters())
        if self.delta is not None:
            params.append(self.delta)
        return params

    # --------------------------------------------------------------- forward
    def pretrained_logits(self, x: torch.Tensor) -> torch.Tensor:
        """``f_P(f_in(x))``: logits over the pre-trained label space."""
        return self.bundle.logits(self.input_transform(x))

    def target_logits(self, x: torch.Tensor, mapping: LabelMapping) -> torch.Tensor:
        """Apply the label mapping ``f_out`` and return target-space logits."""
        logits = self.pretrained_logits(x)
        selected = logits.index_select(1, mapping.source.to(logits.device))
        out = torch.full(
            (logits.shape[0], self.num_target_classes),
            float("-inf"),
            device=logits.device,
            dtype=logits.dtype,
        )
        out[:, mapping.target.to(logits.device)] = selected
        return out


def _device_of(cfg: TrainConfig) -> torch.device:
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if cfg.device == "mps" and not torch.backends.mps.is_available():  # pragma: no cover
        return torch.device("cpu")
    return torch.device(cfg.device)


def build_loaders(cfg: TrainConfig, image_size: int) -> Tuple[DataLoader, DataLoader]:
    batch_size = cfg.resolved_batch_size()
    train_set = build_dataset(
        cfg.dataset, "train", image_size, root=cfg.data_root,
        download=cfg.download, split_dir=cfg.split_dir, seed=cfg.seed,
    )
    test_set = build_dataset(
        cfg.dataset, "test", image_size, root=cfg.data_root,
        download=cfg.download, split_dir=cfg.split_dir, seed=cfg.seed,
    )
    if cfg.train_subset:
        train_set = _subset(train_set, cfg.train_subset, cfg.seed)
    if cfg.test_subset:
        test_set = _subset(test_set, cfg.test_subset, cfg.seed + 1)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=max(batch_size, 256), shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=False,
    )
    return train_loader, test_loader


def _subset(dataset, size: int, seed: int):
    """Deterministic random subset (used by the fast sanity-check runs)."""
    from torch.utils.data import Subset

    size = min(int(size), len(dataset))
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:size].tolist()
    return Subset(dataset, indices)


@torch.no_grad()
def collect_pretrained_predictions(
    model: ReprogramModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Predictions of the *current* ``f_P(f_in(x))`` for the frequency matrix."""
    model.eval()
    preds: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        logits = model.pretrained_logits(x)
        preds.append(logits.argmax(dim=1).detach().cpu())
        labels.append(y.detach().cpu())
    return preds, labels


@torch.no_grad()
def evaluate(
    model: ReprogramModel,
    loader: DataLoader,
    mapping: LabelMapping,
    device: torch.device,
) -> Dict[str, float]:
    """Top-1 accuracy under the label mapping ``f_out``."""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model.target_logits(x, mapping)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return {"accuracy": 100.0 * correct / max(total, 1), "loss": loss_sum / max(total, 1), "n": total}


@torch.no_grad()
def masks_statistics(model: ReprogramModel, loader: DataLoader, device: torch.device,
                     num_batches: int = 4) -> Dict[str, float]:
    """Simple statistics of the learned masks (used for analysis/plots)."""
    if model.input_transform.mask_net is None:
        return {}
    model.eval()
    means, maxes = [], []
    for i, (x, _) in enumerate(loader):
        if i >= num_batches:
            break
        x = x.to(device)
        _, mask = model.input_transform(x, return_mask=True)
        means.append(mask.mean().item())
        maxes.append(mask.max().item())
    return {
        "mask_mean": float(sum(means) / max(len(means), 1)),
        "mask_max": float(max(maxes) if maxes else 0.0),
        "delta_abs_mean": float(model.delta.abs().mean().item()) if model.delta is not None else 0.0,
    }


def train(cfg: TrainConfig, train_loader: DataLoader = None, test_loader: DataLoader = None) -> Dict:
    """Reproduce Algorithm 1 for one (dataset, backbone, method, seed)."""
    torch.manual_seed(cfg.seed)
    device = _device_of(cfg)
    image_size = cfg.resolved_image_size()
    spec = DATASET_SPECS[cfg.dataset]

    model = ReprogramModel(
        cfg.backbone, cfg.dataset, method=cfg.method, image_size=image_size,
        num_layers=cfg.num_layers, num_pool_layers=cfg.num_pool_layers,
        pretrained=cfg.pretrained, device=str(device),
        arch=cfg.arch, pad_width=cfg.pad_width, border_width=cfg.border_width,
        medium_width=cfg.medium_width, hidden_channels=cfg.hidden_channels,
    ).to(device)
    for p in model.backbone.parameters():  # the pre-trained model is never edited
        p.requires_grad_(False)

    if train_loader is None or test_loader is None:
        train_loader, test_loader = build_loaders(cfg, model.image_size)

    # ------------------------------------------------------------- mapping
    if cfg.label_mapping == "rlm":
        mapping = random_label_mapping(
            model.num_pretrained_classes, model.num_target_classes, seed=cfg.seed, device=device
        )
    elif cfg.label_mapping == "flm":
        preds, labels = collect_pretrained_predictions(model, train_loader, device)
        mapping = frequent_label_mapping(
            preds, labels, model.num_pretrained_classes, model.num_target_classes, device
        )
    else:  # ilm: initialised with the identity f_in (epoch 0)
        preds, labels = collect_pretrained_predictions(model, train_loader, device)
        mapping = greedy_injective_mapping(
            frequency_distribution(
                preds, labels, model.num_pretrained_classes, model.num_target_classes, device
            ),
            model.num_target_classes,
            name="ilm",
            device=device,
        )

    # ---------------------------------------------------------- optimisers
    optimizers = []
    schedulers = []
    if model.delta is not None:
        opt_delta = torch.optim.SGD(
            [model.delta], lr=cfg.lr_delta, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
        optimizers.append(opt_delta)
        schedulers.append(
            torch.optim.lr_scheduler.MultiStepLR(
                opt_delta, milestones=list(cfg.milestones), gamma=cfg.gamma_delta
            )
        )
    mask_params = [p for p in model.mask_parameters() if p.requires_grad]
    if mask_params:
        opt_mask = torch.optim.SGD(
            mask_params, lr=cfg.lr_mask, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
        optimizers.append(opt_mask)
        schedulers.append(
            torch.optim.lr_scheduler.MultiStepLR(
                opt_mask, milestones=list(cfg.milestones), gamma=cfg.gamma_mask
            )
        )
    if not optimizers:
        raise ValueError("no trainable parameters: check the method configuration")

    history: List[Dict] = []
    start = time.time()
    for epoch in range(cfg.epochs):
        # ---------- Algorithm 4 step: refresh f_out before training the epoch
        if cfg.label_mapping == "ilm" and epoch > 0 and (epoch % cfg.ilm_every_n_epochs == 0):
            preds, labels = collect_pretrained_predictions(model, train_loader, device)
            mapping = greedy_injective_mapping(
                frequency_distribution(
                    preds, labels, model.num_pretrained_classes, model.num_target_classes, device
                ),
                model.num_target_classes,
                name="ilm",
                device=device,
            )

        model.train()
        model.bundle.model.eval()
        running_loss, seen, correct = 0.0, 0, 0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model.target_logits(x, mapping)
            loss = F.cross_entropy(logits, y)

            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in optimizers:
                opt.step()

            running_loss += float(loss.item()) * y.numel()
            seen += int(y.numel())
            correct += int((logits.argmax(dim=1) == y).sum().item())

        for sched in schedulers:
            sched.step()

        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "train_accuracy": 100.0 * correct / max(seen, 1),
            "lr_delta": optimizers[0].param_groups[0]["lr"],
        }
        if cfg.eval_every and (epoch + 1) % cfg.eval_every == 0:
            metrics = evaluate(model, test_loader, mapping, device)
            record.update({"test_accuracy": metrics["accuracy"], "test_loss": metrics["loss"]})
        if cfg.save_masks_every and (epoch + 1) % cfg.save_masks_every == 0:
            record.update(masks_statistics(model, train_loader, device))
        history.append(record)

    final = evaluate(model, test_loader, mapping, device)
    result = {
        "config": {**asdict(cfg), "milestones": list(cfg.milestones)},
        "image_size": model.image_size,
        "dataset": cfg.dataset,
        "num_classes": spec.num_classes,
        "method": cfg.method,
        "label_mapping": cfg.label_mapping,
        "seed": cfg.seed,
        "test_accuracy": final["accuracy"],
        "test_loss": final["loss"],
        "test_size": final["n"],
        "num_mask_parameters": int(sum(p.numel() for p in model.mask_parameters())),
        "num_delta_parameters": int(model.delta.numel()) if model.delta is not None else 0,
        "label_mapping_table": mapping.as_dict(),
        "history": history,
        "wall_time_sec": time.time() - start,
    }
    return result


def save_result(result: Dict, cfg: TrainConfig) -> str:
    """Persist the run under ``<output_dir>/<dataset>/<method>_seed<seed>.json``."""
    run_name = cfg.run_name or f"{cfg.backbone}_{cfg.dataset}_{cfg.method}_{cfg.label_mapping}_seed{cfg.seed}"
    out_dir = os.path.join(cfg.output_dir, cfg.dataset)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run_name}.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    return path
