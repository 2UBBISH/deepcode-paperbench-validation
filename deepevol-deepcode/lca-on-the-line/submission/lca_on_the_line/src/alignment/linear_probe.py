"""Linear probing with taxonomy-aligned soft labels (paper §4.3.2, §E.2, §E.3, §E.5).

This module implements the linear-probe training procedure used in Section 4.3.2 of
*LCA-on-the-Line*.  A single linear layer (``nn.Linear(feature_dim, num_classes)``) is
trained on top of frozen penultimate features ``M(X)`` of a pretrained backbone, using
either

* ``CE`` only — the *Baseline* rows of Table 5 / Table 9, i.e. the standard cross-entropy
  loss; or
* ``CE + LCA soft loss`` — the *Ours* rows, i.e. Algorithm 1 of Appendix E.2:
  :math:`L = \\lambda L(\\mathrm{CE}) + L(\\mathrm{soft}_{lca})` with soft multi-label
  targets given by the rows of ``reverse_LCA_matrix = 1 - MinMax(M ** T)``.

For every backbone we additionally train the *interpolation* variant of Wortsman et al.
(2022) used as the final classifier in the paper:

.. math::

    W_{\\text{interp}} = \\alpha W_{ce} + (1 - \\alpha) W_{ce + \\text{soft}}

where :math:`\\alpha` is swept over a grid and the operating points are selected exactly
as described in Table 9:

* *no ID accuracy drop* — largest :math:`\\alpha` (closest to the CE-only weights) whose ID
  accuracy still matches the CE-only baseline; and
* *pro-OOD* — smallest :math:`\\alpha` (closest to the soft-label weights).

Hyperparameters follow Appendix E.5 verbatim: learning rate ``0.001``, batch size ``1024``,
AdamW with weight decay, cosine schedule with a *linear* warm-up of learning rate ``1e-5``,
``50`` epochs.  Soft-loss defaults follow Appendix E.2: :math:`\\lambda = 0.03`,
temperature ``25`` and CE alignment mode.  Where the paper is silent we use sensible,
documented defaults (see the module constants below).

The module is torch-optional and numpy-optional at import time (heavy imports are lazy) so
that the training configuration can be inspected in minimal environments.
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Defaults — Appendix E.5 (hyperparameters) and Appendix E.2 (soft loss)
# --------------------------------------------------------------------------------------
DEFAULT_LEARNING_RATE = 0.001          # §E.5
DEFAULT_BATCH_SIZE = 1024              # §E.5
DEFAULT_EPOCHS = 50                    # §E.5
DEFAULT_WEIGHT_DECAY = 0.05            # §E.5 says "AdamW with weight decay" (value not given)
DEFAULT_WARMUP_TYPE = "linear"         # §E.5
DEFAULT_WARMUP_LR = 1e-5               # §E.5
DEFAULT_WARMUP_RATIO = 0.05            # §E.5 says "a warm-up iteration" (fraction not given)
DEFAULT_SCHEDULER = "cosine"           # §E.5
DEFAULT_LAMBDA_WEIGHT = 0.03           # §E.2 / Algorithm 1
DEFAULT_TEMPERATURE = 25.0             # §E.2
DEFAULT_ALIGNMENT_MODE = "CE"          # §E.2 ("use CE as the soft loss")
DEFAULT_NUM_WORKERS = 4
DEFAULT_SEED = 0
DEFAULT_INTERP_GRID: Tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(11))  # 0.0 … 1.0
DEFAULT_PROBE_EPOCHS_TOLERANCE = 0.0   # "without compromising ID performance" (strict by default)


def _require_torch():
    """Lazily import torch (keeps the module importable in minimal environments)."""
    try:
        import torch  # noqa: WPS433 (local import is intentional)
        import torch.nn as nn  # noqa: WPS433
        import torch.nn.functional as F  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "linear_probe requires PyTorch. Install it with `pip install torch`."
        ) from exc
    return torch, nn, F


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Best-effort deterministic seeding for torch / numpy / python random."""
    random.seed(seed)
    try:  # pragma: no cover - optional
        import numpy as np

        np.random.seed(seed % (2 ** 32))
    except Exception:
        pass
    try:  # pragma: no cover - optional
        torch, _, _ = _require_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class ProbeConfig:
    """Training configuration for a single linear probe (Appendix E.5)."""

    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    optimizer: str = "adamw"
    scheduler: str = DEFAULT_SCHEDULER
    warmup_type: str = DEFAULT_WARMUP_TYPE
    warmup_lr: float = DEFAULT_WARMUP_LR
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    warmup_steps: Optional[int] = None
    lambda_weight: float = DEFAULT_LAMBDA_WEIGHT
    temperature: float = DEFAULT_TEMPERATURE
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE
    use_soft_loss: bool = True
    seed: int = DEFAULT_SEED
    num_workers: int = DEFAULT_NUM_WORKERS
    device: Optional[str] = None
    log_every: int = 0
    max_steps: Optional[int] = None  # for smoke tests
    feature_dim: Optional[int] = None
    num_classes: Optional[int] = None
    eval_datasets: Optional[Sequence[str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "ProbeConfig":
        """Build a config from a (possibly partial, possibly nested) mapping."""
        payload = dict(payload or {})
        if not payload:
            return cls()
        # Accept nested sections such as {"linear_probe": {...}} or {"soft_loss": {...}}.
        nested = {}
        for key in ("linear_probe", "probe", "soft_loss", "align"):
            section = payload.pop(key, None)
            if isinstance(section, dict):
                nested.update(section)
        nested.update(payload)
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in nested.items() if k in valid}
        unknown = {k: v for k, v in nested.items() if k not in valid}
        config = cls(**kwargs)
        if unknown:
            config.extra.update(unknown)
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# --------------------------------------------------------------------------------------
# Cosine schedule with linear warm-up (§E.5)
# --------------------------------------------------------------------------------------
def build_linear_warmup_cosine_scheduler(
    optimizer: Any,
    total_steps: int,
    warmup_steps: Optional[int] = None,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    warmup_type: str = DEFAULT_WARMUP_TYPE,
    warmup_lr: float = DEFAULT_WARMUP_LR,
    base_lr: Optional[float] = None,
    scheduler: str = DEFAULT_SCHEDULER,
) -> Any:
    """AdamW-compatible cosine schedule with a linear warm-up (as reported in §E.5).

    The learning rate rises linearly from ``warmup_lr`` to the optimizer's base LR over
    ``warmup_steps`` iterations and then follows a cosine decay to zero.  Implemented with
    a ``LambdaLR`` so it works for any optimizer and is trivial to inspect.
    """
    torch, _, _ = _require_torch()
    from torch.optim.lr_scheduler import LambdaLR

    if base_lr is None:
        base_lr = optimizer.param_groups[0]["lr"]
    if warmup_steps is None:
        warmup_steps = int(round(max(1.0, warmup_ratio * float(max(1, total_steps)))))
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    start_factor = min(1.0, float(warmup_lr) / float(base_lr)) if base_lr else 0.0

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            if warmup_type == "linear":
                progress = float(step) / float(warmup_steps)
                return start_factor + (1.0 - start_factor) * progress
            return 1.0  # constant warm-up
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        if scheduler in ("cosine", "cosineannealinglr"):
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if scheduler in ("linear", "linear_decay"):
            return 1.0 - progress
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def build_optimizer(parameters: Iterable[Any], config: Optional[ProbeConfig] = None) -> Any:
    """AdamW optimizer with the §E.5 learning rate / weight decay."""
    torch, _, _ = _require_torch()
    cfg = config or ProbeConfig()
    params = [p for p in parameters if p.requires_grad]
    name = (cfg.optimizer or "adamw").lower()
    if name in ("adamw", "adam_w"):
        return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if name in ("sgd", "momentum"):
        return torch.optim.SGD(
            params, lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay
        )
    raise ValueError(f"Unknown optimizer '{cfg.optimizer}'")


# --------------------------------------------------------------------------------------
# Feature / target containers
# --------------------------------------------------------------------------------------
@dataclass
class ProbeData:
    """Frozen features + labels (and optionally labels for other splits)."""

    features: Any  # (N, D) tensor or ndarray
    targets: Any   # (N,) tensor or ndarray
    name: str = "id"


def _to_tensor(tensor_like: Any, dtype: Any) -> Any:
    torch, _, _ = _require_torch()
    if tensor_like is None:
        return None
    if hasattr(tensor_like, "detach"):  # already a torch tensor
        return tensor_like.detach().to(dtype)
    try:
        import numpy as np

        return torch.as_tensor(np.asarray(tensor_like), dtype=dtype)
    except Exception:  # pragma: no cover - defensive
        return torch.as_tensor(tensor_like, dtype=dtype)


def _as_numpy(tensor_like: Any):
    import numpy as np

    if tensor_like is None:
        return None
    if hasattr(tensor_like, "detach"):
        return tensor_like.detach().cpu().numpy()
    return np.asarray(tensor_like)


def make_labeled_folds(
    features: Any,
    targets: Any,
    fold_masks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[Any, Any]]:
    """Split features/targets into named folds (``{"train": ..., "val": ...}``)."""
    folds: Dict[str, Tuple[Any, Any]] = {}
    if fold_masks is None:
        folds["train"] = (features, targets)
        return folds
    for name, mask in fold_masks.items():
        import numpy as np

        m = np.asarray(mask)
        if hasattr(features, "detach"):
            feats = features[m]
            tars = targets[m]
        else:
            feats = np.asarray(features)[m]
            tars = np.asarray(targets)[m]
        folds[name] = (feats, tars)
    return folds


def make_feature_loader(
    features: Any,
    targets: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    seed: int = DEFAULT_SEED,
    num_workers: int = 0,
    drop_last: bool = False,
) -> Any:
    """TensorDataset/DataLoader over frozen features (no image decoding involved)."""
    torch, _, _ = _require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    feats = _to_tensor(features, torch.float32)
    tars = _to_tensor(targets, torch.long)
    if feats.dim() > 2:
        feats = feats.flatten(1)
    dataset = TensorDataset(feats, tars)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        drop_last=bool(drop_last),
        generator=generator if shuffle else None,
    )


# --------------------------------------------------------------------------------------
# The probe itself
# --------------------------------------------------------------------------------------
def build_linear_probe(
    feature_dim: int,
    num_classes: int,
    normalize_features: bool = True,
    bias: bool = True,
    device: Optional[str] = None,
) -> Any:
    """A single ``nn.Linear(feature_dim, num_classes)`` layer.

    ``normalize_features=True`` adds an L2-normalisation of the frozen features before the
    linear layer, which makes the probe's weights comparable across backbones (this is a
    no-op w.r.t. the paper's formulation, since it is a fixed transform of ``M(X)``).
    """
    torch, nn, _ = _require_torch()
    layers: List[Any] = []
    if normalize_features:
        layers.append(_L2Normalize())
    layers.append(nn.Linear(int(feature_dim), int(num_classes), bias=bias))
    model = nn.Sequential(*layers) if len(layers) > 1 else layers[0]
    if device is not None:
        model = model.to(device)
    return model


class _L2Normalize(object):
    """Picklable L2-normalisation module (views the probe input as features)."""

    def __new__(cls, eps: float = 1e-8):
        torch, nn, F = _require_torch()

        class _L2(nn.Module):
            def __init__(self, eps: float = 1e-8):
                super().__init__()
                self.eps = eps

            def forward(self, x):
                return F.normalize(x, dim=-1, eps=self.eps)

        return _L2(eps)


def get_probe_weights(probe: Any) -> Any:
    """Return the ``(num_classes, feature_dim)`` weight matrix of a probe."""
    _, nn, _ = _require_torch()
    if isinstance(probe, nn.Sequential):
        return probe[-1].weight
    return probe.weight


def get_probe_bias(probe: Any) -> Any:
    _, nn, _ = _require_torch()
    if isinstance(probe, nn.Sequential):
        return probe[-1].bias
    return probe.bias


def set_probe_weights(probe: Any, weight: Any, bias: Any = None) -> None:
    """In-place assignment of the linear-layer parameters."""
    import numpy as np

    target = probe[-1] if hasattr(probe, "__getitem__") and not hasattr(probe, "weight") else probe
    with _no_grad():
        target.weight.copy_(_to_tensor(weight, target.weight.dtype))
        if bias is not None and target.bias is not None:
            target.bias.copy_(_to_tensor(bias, target.bias.dtype))


def _no_grad():
    torch, _, _ = _require_torch()
    return torch.no_grad()


# --------------------------------------------------------------------------------------
# Loss construction (Algorithm 1 of §E.2)
# --------------------------------------------------------------------------------------
def build_soft_loss(
    lca_matrix: Any,
    lambda_weight: float = DEFAULT_LAMBDA_WEIGHT,
    temperature: float = DEFAULT_TEMPERATURE,
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    num_classes: Optional[int] = None,
    device: Optional[str] = None,
    processed: bool = True,
    use_soft_loss: bool = True,
) -> Any:
    """Factory for :class:`LCAAlignmentLoss` from ``src.alignment.soft_loss``.

    ``lca_matrix`` may be:

    * a raw ``n x n`` LCA *distance* matrix (then ``M_LCA = MinMax(M ** T)`` is applied
      locally, per §E.2), or
    * a pre-processed ``M_LCA`` when ``processed=True`` and its values already lie in
      ``[0, 1]`` (the ``process_lca_matrix`` pipeline of ``src.hierarchy.lca_matrix``).
    """
    from .soft_loss import LCAAlignmentLoss, process_lca_matrix  # local import (lazy torch)

    matrix = lca_matrix
    if processed:
        # Detect an obviously unprocessed (unscaled) matrix and process it here.
        values = _as_numpy(matrix)
        is_unit_range = values is not None and values.size and values.min() >= -1e-6 and values.max() <= 1.0 + 1e-6
        has_unit_diagonal = values is not None and values.size and abs(float(values[0, 0]) - 1.0) < 1e-6
        if not (is_unit_range and has_unit_diagonal):
            matrix = process_lca_matrix(
                lca_matrix, temperature=temperature, as_tensor=False, latent_hierarchy=False
            )
    return LCAAlignmentLoss(
        lca_matrix=matrix,
        lambda_weight=lambda_weight,
        alignment_mode=alignment_mode,
        temperature=temperature,
        processed_matrix=True,
        num_classes=num_classes,
        device=device,
        enabled=use_soft_loss,
    )


# --------------------------------------------------------------------------------------
# Accuracy evaluation
# --------------------------------------------------------------------------------------
def probe_logits(probe: Any, features: Any, batch_size: int = DEFAULT_BATCH_SIZE, device: Any = None) -> Any:
    """Forward the probe over frozen features in chunks (eval mode, no grad)."""
    torch, _, _ = _require_torch()
    probe.eval()
    chunks: List[Any] = []
    with torch.no_grad():
        for start in range(0, len(features), int(batch_size)):
            batch = _to_tensor(features[start : start + int(batch_size)], torch.float32)
            if device is not None:
                batch = batch.to(device)
            if batch.dim() > 2:
                batch = batch.flatten(1)
            out = probe(batch)
            chunks.append(out.detach().cpu())
    if not chunks:
        return torch.zeros((0, 0))
    return torch.cat(chunks, dim=0)


def probe_accuracy(probe: Any, features: Any, targets: Any, batch_size: int = DEFAULT_BATCH_SIZE, device: Any = None) -> float:
    """Top-1 accuracy (%) of a probe on frozen features."""
    logits = probe_logits(probe, features, batch_size=batch_size, device=device)
    if logits.numel() == 0:
        return float("nan")
    tars = _to_tensor(targets, _require_torch()[0].long)
    preds = logits.argmax(dim=1)
    correct = (preds == tars).float().mean().item()
    return 100.0 * correct


def probe_top5_accuracy(probe: Any, features: Any, targets: Any, batch_size: int = DEFAULT_BATCH_SIZE, device: Any = None) -> float:
    """Top-5 accuracy (%) of a probe on frozen features."""
    torch, _, _ = _require_torch()
    logits = probe_logits(probe, features, batch_size=batch_size, device=device)
    if logits.numel() == 0:
        return float("nan")
    tars = _to_tensor(targets, torch.long)
    k = min(5, logits.shape[1])
    topk = logits.topk(k, dim=1).indices
    correct = (topk == tars.view(-1, 1)).any(dim=1).float().mean().item()
    return 100.0 * correct


def evaluate_probe(
    probe: Any,
    datasets: Dict[str, Union[ProbeData, Tuple[Any, Any]]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Any = None,
    compute_top5: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a probe on several frozen-feature splits -> ``{name: {top1, top5, n}}``."""
    results: Dict[str, Dict[str, float]] = {}
    for name, payload in datasets.items():
        feats, tars = (payload.features, payload.targets) if isinstance(payload, ProbeData) else payload
        row = {
            "top1": probe_accuracy(probe, feats, tars, batch_size=batch_size, device=device),
            "n": float(len(_as_numpy(tars))) if tars is not None else 0.0,
        }
        if compute_top5:
            row["top5"] = probe_top5_accuracy(probe, feats, tars, batch_size=batch_size, device=device)
        results[name] = row
    return results


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
@dataclass
class ProbeTrainResult:
    """Artifacts of one probe training run."""

    probe: Any
    mode: str
    history: List[Dict[str, float]] = field(default_factory=list)
    config: Optional[ProbeConfig] = None
    id_accuracy: float = float("nan")
    soft_loss_enabled: bool = False

    def accuracy(self) -> float:
        return self.id_accuracy


def train_linear_probe(
    train_features: Any,
    train_targets: Any,
    num_classes: int,
    lca_matrix: Any = None,
    config: Optional[ProbeConfig] = None,
    use_soft_loss: bool = True,
    feature_dim: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = False,
    on_epoch: Optional[Callable[[int, Any], None]] = None,
) -> ProbeTrainResult:
    """Train a single linear probe on frozen features (Algorithm 1 + §E.5 schedule).

    Parameters
    ----------
    train_features / train_targets:
        Frozen penultimate features ``M(X)`` and ImageNet labels of the *probing* dataset
        (ImageNet-1k train split in the paper).
    num_classes:
        Size of the classification head (1000 for ImageNet-1k).
    lca_matrix:
        Raw ``n x n`` pairwise LCA distance matrix.  When ``None`` or
        ``use_soft_loss=False`` the probe is trained with cross-entropy only (Baseline).
    config:
        :class:`ProbeConfig`; defaults to the §E.5 hyperparameters.
    """
    torch, nn, F = _require_torch()
    cfg = config or ProbeConfig()
    set_seed(cfg.seed)
    dev = torch.device(device or cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    feats = _to_tensor(train_features, torch.float32)
    tars = _to_tensor(train_targets, torch.long)
    if feats.dim() > 2:
        feats = feats.flatten(1)
    dim = int(feature_dim or feats.shape[1])

    probe = build_linear_probe(dim, num_classes, normalize_features=True, device=dev)
    optimizer = build_optimizer(probe.parameters(), cfg)

    loader = make_feature_loader(
        feats, tars, batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed, num_workers=0
    )
    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * int(cfg.epochs)
    if cfg.max_steps is not None:
        total_steps = min(total_steps, int(cfg.max_steps))
    scheduler = build_linear_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=cfg.warmup_steps,
        warmup_ratio=cfg.warmup_ratio,
        warmup_type=cfg.warmup_type,
        warmup_lr=cfg.warmup_lr,
        base_lr=cfg.learning_rate,
        scheduler=cfg.scheduler,
    )

    soft_available = bool(use_soft_loss) and lca_matrix is not None
    criterion = None
    if soft_available:
        from .soft_loss import LCAAlignmentLoss

        criterion = LCAAlignmentLoss(
            lca_matrix=lca_matrix,
            lambda_weight=cfg.lambda_weight,
            alignment_mode=cfg.alignment_mode,
            temperature=cfg.temperature,
            num_classes=int(num_classes),
            device=dev,
        )

    history: List[Dict[str, float]] = []
    global_step = 0
    probe.train()
    stop = False
    for epoch in range(int(cfg.epochs)):
        running = 0.0
        n_seen = 0
        for feats_b, tars_b in loader:
            feats_b = feats_b.to(dev)
            tars_b = tars_b.to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits = probe(feats_b)
            if soft_available:
                loss = criterion(logits, tars_b)
            else:
                loss = F.cross_entropy(logits, tars_b)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            running += float(loss.detach().item()) * feats_b.shape[0]
            n_seen += feats_b.shape[0]
            if cfg.max_steps is not None and global_step >= int(cfg.max_steps):
                stop = True
                break
        row = {
            "epoch": float(epoch),
            "loss": running / max(1, n_seen),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if verbose or (cfg.log_every and (epoch % cfg.log_every == 0)):
            LOG.info("epoch %d: loss=%.4f lr=%.2e", epoch, row["loss"], row["lr"])
        history.append(row)
        if on_epoch is not None:
            on_epoch(epoch, probe)
        if stop:
            break

    id_acc = probe_accuracy(probe, feats, tars, batch_size=cfg.batch_size, device=dev)
    return ProbeTrainResult(
        probe=probe,
        mode="ce+soft" if soft_available else "ce",
        history=history,
        config=cfg,
        id_accuracy=id_acc,
        soft_loss_enabled=soft_available,
    )


# --------------------------------------------------------------------------------------
# Weight-space interpolation (Wortsman et al., 2022) — §4.3.2 and Table 9
# --------------------------------------------------------------------------------------
@dataclass
class InterpolationPoint:
    """One :math:`\\alpha` operating point of the interpolation sweep."""

    alpha: float
    id_accuracy: float
    ood_accuracy: Dict[str, float] = field(default_factory=dict)
    mean_ood: float = float("nan")

    def asdict(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {"alpha": self.alpha, "id_top1": self.id_accuracy}
        for key, value in self.ood_accuracy.items():
            row[f"ood_top1_{key}"] = value
        row["mean_ood"] = self.mean_ood
        return row


@dataclass
class InterpolationResult:
    """Full result of a weight-interpolation sweep, incl. the Table 9 selections."""

    alpha_grid: List[float]
    points: List[InterpolationPoint]
    best_no_id_drop: Optional[InterpolationPoint] = None
    best_pro_ood: Optional[InterpolationPoint] = None
    ce_id_accuracy: float = float("nan")

    def asdict(self) -> Dict[str, Any]:
        return {
            "alpha_grid": list(self.alpha_grid),
            "ce_id_accuracy": self.ce_id_accuracy,
            "points": [p.asdict() for p in self.points],
            "no_id_drop": self.best_no_id_drop.asdict() if self.best_no_id_drop else None,
            "pro_ood": self.best_pro_ood.asdict() if self.best_pro_ood else None,
        }


def interpolate_probes(ce_probe: Any, soft_probe: Any, alpha: float, inplace: bool = False) -> Any:
    """``W_interp = alpha * W_ce + (1 - alpha) * W_ce+soft`` (§4.3.2).

    Interpolates both the weight matrix and (when present) the bias of the linear layer.
    ``alpha = 1`` recovers the CE-only probe, ``alpha = 0`` the soft-label probe.
    """
    torch, nn, _ = _require_torch()
    if not 0.0 - 1e-9 <= float(alpha) <= 1.0 + 1e-9:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    import copy

    target = ce_probe if inplace else copy.deepcopy(ce_probe)
    w_ce, w_soft = get_probe_weights(ce_probe), get_probe_weights(soft_probe)
    b_ce, b_soft = get_probe_bias(ce_probe), get_probe_bias(soft_probe)
    a = float(alpha)
    with torch.no_grad():
        w_mix = a * w_ce.detach() + (1.0 - a) * w_soft.detach().to(w_ce.device)
        set_probe_weights(target, w_mix, None)
        if b_ce is not None and b_soft is not None:
            b_mix = a * b_ce.detach() + (1.0 - a) * b_soft.detach().to(b_ce.device)
            set_probe_weights(target, w_mix, b_mix)
    return target


def interpolation_grid(grid: Optional[Sequence[float]] = None) -> List[float]:
    """Default alpha grid ``0.0 … 1.0`` (step 0.1), or ``grid`` if given."""
    if grid is None:
        return list(DEFAULT_INTERP_GRID)
    return [float(x) for x in grid]


def interpolate_and_evaluate(
    ce_probe: Any,
    soft_probe: Any,
    id_data: Union[ProbeData, Tuple[Any, Any]],
    ood_data: Optional[Dict[str, Union[ProbeData, Tuple[Any, Any]]]] = None,
    grid: Optional[Sequence[float]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Any = None,
    id_tolerance: float = DEFAULT_PROBE_EPOCHS_TOLERANCE,
    ce_id_accuracy: Optional[float] = None,
    verbose: bool = False,
) -> InterpolationResult:
    """Sweep :math:`\\alpha` and select the two Table-9 operating points.

    * **no ID accuracy drop** — the operating point closest to CE-only (largest
      :math:`\\alpha`) whose ID accuracy is at least ``ce_id_accuracy - id_tolerance``.
    * **pro-OOD** — the operating point with the best mean OOD accuracy, i.e. the smallest
      :math:`\\alpha` (closest to the soft-label probe) that does not hurt OOD.
    """
    id_feats, id_tars = (id_data.features, id_data.targets) if isinstance(id_data, ProbeData) else id_data
    soft_id_acc = (
        float(ce_id_accuracy)
        if ce_id_accuracy is not None
        else probe_accuracy(ce_probe, id_feats, id_tars, batch_size=batch_size, device=device)
    )
    points: List[InterpolationPoint] = []
    for alpha in interpolation_grid(grid):
        mixed = interpolate_probes(ce_probe, soft_probe, alpha)
        id_acc = probe_accuracy(mixed, id_feats, id_tars, batch_size=batch_size, device=device)
        ood_acc: Dict[str, float] = {}
        if ood_data:
            for name, payload in ood_data.items():
                f, t = (payload.features, payload.targets) if isinstance(payload, ProbeData) else payload
                ood_acc[name] = probe_accuracy(mixed, f, t, batch_size=batch_size, device=device)
        mean_ood = (
            sum(ood_acc.values()) / len(ood_acc) if ood_acc else float("nan")
        )
        points.append(
            InterpolationPoint(alpha=alpha, id_accuracy=id_acc, ood_accuracy=ood_acc, mean_ood=mean_ood)
        )

    # "No ID accuracy drop": largest alpha still within tolerance of the CE-only accuracy.
    no_drop_candidates = [p for p in points if p.id_accuracy >= soft_id_acc - float(id_tolerance)]
    best_no_drop = max(no_drop_candidates, key=lambda p: p.alpha) if no_drop_candidates else None

    # "Pro-OOD": best mean OOD accuracy; ties broken towards larger alpha (closer to CE).
    pro_ood_candidates = [p for p in points if not math.isnan(p.mean_ood)]
    best_pro_ood = (
        max(pro_ood_candidates, key=lambda p: (p.mean_ood, p.alpha))
        if pro_ood_candidates
        else None
    )
    result = InterpolationResult(
        alpha_grid=[p.alpha for p in points],
        points=points,
        best_no_id_drop=best_no_drop,
        best_pro_ood=best_pro_ood,
        ce_id_accuracy=soft_id_acc,
    )
    if verbose:
        LOG.info(
            "interpolation: ce_id=%.2f | no-drop alpha=%s id=%.2f | pro-OOD alpha=%s mean_ood=%.2f",
            soft_id_acc,
            getattr(best_no_drop, "alpha", None),
            getattr(best_no_drop, "id_accuracy", float("nan")),
            getattr(best_pro_ood, "alpha", None),
            getattr(best_pro_ood, "mean_ood", float("nan")),
        )
    return result


# --------------------------------------------------------------------------------------
# End-to-end: train CE and CE+soft probes and interpolate (Table 5 / 6 / 9 / 10)
# --------------------------------------------------------------------------------------
@dataclass
class ProbeExperimentResult:
    """Result bundle for one backbone × one hierarchy source (Table 5 / 6 / 9)."""

    backbone: str
    hierarchy_source: str
    baseline: Dict[str, Dict[str, float]] = field(default_factory=dict)
    soft: Dict[str, Dict[str, float]] = field(default_factory=dict)
    interpolation: Optional[InterpolationResult] = None
    no_id_drop: Optional[Dict[str, Any]] = None
    pro_ood: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None

    def asdict(self) -> Dict[str, Any]:
        return {
            "backbone": self.backbone,
            "hierarchy_source": self.hierarchy_source,
            "baseline": self.baseline,
            "soft": self.soft,
            "interpolation": self.interpolation.asdict() if self.interpolation else None,
            "no_id_drop": self.no_id_drop,
            "pro_ood": self.pro_ood,
            "config": self.config,
        }


def run_linear_probe_experiment(
    train_features: Any,
    train_targets: Any,
    id_eval: Union[ProbeData, Tuple[Any, Any]],
    ood_eval: Optional[Dict[str, Union[ProbeData, Tuple[Any, Any]]]] = None,
    lca_matrix: Any = None,
    num_classes: int = 1000,
    config: Optional[ProbeConfig] = None,
    backbone: str = "resnet18",
    hierarchy_source: str = "wordnet",
    grid: Optional[Sequence[float]] = None,
    device: Optional[str] = None,
    verbose: bool = False,
    ce_probe: Any = None,
    soft_probe: Any = None,
) -> ProbeExperimentResult:
    """Train Baseline + soft-label probes and produce the *Ours* (interpolated) numbers.

    Mirrors the paper's protocol: train a CE-only probe and a CE + LCA-soft-loss probe with
    identical hyperparameters (§E.5), linearly interpolate their weights (§4.3.2), then report
    (i) the *no ID accuracy drop* and (ii) the *pro-OOD* operating points of Table 9.
    """
    cfg = config or ProbeConfig()
    if ce_probe is None:
        ce_res = train_linear_probe(
            train_features, train_targets, num_classes, lca_matrix=None,
            config=cfg, use_soft_loss=False, device=device, verbose=verbose,
        )
        ce_probe = ce_res.probe
    if soft_probe is None:
        soft_res = train_linear_probe(
            train_features, train_targets, num_classes, lca_matrix=lca_matrix,
            config=cfg, use_soft_loss=True, device=device, verbose=verbose,
        )
        soft_probe = soft_res.probe

    id_payload = id_eval if isinstance(id_eval, ProbeData) else ProbeData(*id_eval, name="id")
    eval_sets: Dict[str, Any] = {"id": id_payload}
    if ood_eval:
        for name, payload in ood_eval.items():
            eval_sets[name] = payload if isinstance(payload, ProbeData) else ProbeData(*payload, name=name)

    baseline = evaluate_probe(ce_probe, eval_sets, batch_size=cfg.batch_size, device=device)
    soft = evaluate_probe(soft_probe, eval_sets, batch_size=cfg.batch_size, device=device)
    interp = interpolate_and_evaluate(
        ce_probe,
        soft_probe,
        id_payload,
        ood_data={k: v for k, v in eval_sets.items() if k != "id"},
        grid=grid,
        batch_size=cfg.batch_size,
        device=device,
        ce_id_accuracy=baseline["id"]["top1"],
        verbose=verbose,
    )
    result = ProbeExperimentResult(
        backbone=backbone,
        hierarchy_source=hierarchy_source,
        baseline=baseline,
        soft=soft,
        interpolation=interp,
        no_id_drop=interp.best_no_id_drop.asdict() if interp.best_no_id_drop else None,
        pro_ood=interp.best_pro_ood.asdict() if interp.best_pro_ood else None,
        config=cfg.to_dict(),
    )
    return result


def build_hierarchy_soft_loss(
    hierarchy: Any,
    temperature: float = DEFAULT_TEMPERATURE,
    lambda_weight: float = DEFAULT_LAMBDA_WEIGHT,
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    num_classes: Optional[int] = None,
    device: Optional[str] = None,
) -> Any:
    """Build the Algorithm-1 loss for either a WordNet or a latent (K-means) hierarchy.

    ``hierarchy`` may be a :class:`~src.hierarchy.wordnet.WordNetHierarchy`, a
    :class:`~src.hierarchy.latent_kmeans.LatentHierarchy`, or a pre-computed distance
    matrix (nested list / ndarray / tensor).
    """
    if isinstance(hierarchy, (list, tuple)):
        matrix = hierarchy
        if matrix and isinstance(matrix[0], (list, tuple)):
            matrix = [list(row) for row in matrix]
        return build_soft_loss(
            matrix,
            lambda_weight=lambda_weight,
            temperature=temperature,
            alignment_mode=alignment_mode,
            num_classes=num_classes,
            device=device,
            processed=False,
        )
    if hasattr(hierarchy, "latent_lca_matrix"):
        matrix = hierarchy.latent_lca_matrix()
        if callable(matrix):
            matrix = matrix()
        return build_soft_loss(
            matrix,
            lambda_weight=lambda_weight,
            temperature=temperature,
            alignment_mode=alignment_mode,
            num_classes=num_classes,
            device=device,
            processed=False,
        )
    if hasattr(hierarchy, "distance_matrix"):
        matrix = hierarchy.distance_matrix()
        if callable(matrix):
            matrix = matrix()
        return build_soft_loss(
            matrix,
            lambda_weight=lambda_weight,
            temperature=temperature,
            alignment_mode=alignment_mode,
            num_classes=num_classes,
            device=device,
            processed=False,
        )
    # A WordNet-style tree: build the raw pairwise information-content matrix.
    from ..hierarchy.lca import pairwise_lca_matrix

    matrix = pairwise_lca_matrix(hierarchy)
    return build_soft_loss(
        matrix,
        lambda_weight=lambda_weight,
        temperature=temperature,
        alignment_mode=alignment_mode,
        num_classes=num_classes,
        device=device,
        processed=False,
    )


# --------------------------------------------------------------------------------------
# Cached-feature convenience (reuses evaluate_models .npz caches)
# --------------------------------------------------------------------------------------
def load_probe_data_from_cache(
    cache_dir: str,
    model_name: str,
    dataset: str,
    data_key: str = "features",
    target_key: str = "targets",
) -> ProbeData:
    """Load frozen features/labels for ``(model_name, dataset)`` from a cached ``.npz``."""
    import numpy as np

    candidates = [
        os.path.join(cache_dir, f"{model_name}__{dataset}.npz"),
        os.path.join(cache_dir, f"{model_name}--{dataset}.npz"),
        os.path.join(cache_dir, "outputs", f"{model_name}__{dataset}.npz"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            f"No cached outputs for model='{model_name}' dataset='{dataset}' in {cache_dir}"
        )
    payload = np.load(path, allow_pickle=True)
    if data_key not in payload:
        raise KeyError(f"Cache {path} has no '{data_key}' array (available: {list(payload.files)})")
    features = payload[data_key]
    targets = payload[target_key] if target_key in payload else None
    return ProbeData(features=features, targets=targets, name=f"{model_name}:{dataset}")


# --------------------------------------------------------------------------------------
# Self-test (no dataset required)
# --------------------------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover - manual smoke test
    """Tiny end-to-end smoke test on synthetic features."""
    import numpy as np

    torch, _, _ = _require_torch()
    rng = np.random.default_rng(0)
    num_classes, dim, per_class = 10, 16, 20
    centers = rng.normal(size=(num_classes, dim)).astype("float32")
    feats = np.concatenate([centers[c] + 0.3 * rng.normal(size=(per_class, dim)) for c in range(num_classes)])
    tars = np.concatenate([np.full(per_class, c) for c in range(num_classes)]).astype("int64")

    # A simple block-structured "LCA distance" matrix.
    matrix = np.abs(np.subtract.outer(np.arange(num_classes), np.arange(num_classes))).astype("float32")
    cfg = ProbeConfig(epochs=2, batch_size=64, max_steps=20)
    ce = train_linear_probe(feats, tars, num_classes, lca_matrix=None, config=cfg)
    soft = train_linear_probe(feats, tars, num_classes, lca_matrix=matrix, config=cfg)
    interp = interpolate_and_evaluate(ce.probe, soft.probe, (feats, tars), grid=(0.0, 0.5, 1.0))
    print(
        f"ce_id={ce.id_accuracy:.2f} soft_id={soft.id_accuracy:.2f} "
        f"no_drop(alpha={interp.best_no_id_drop.alpha if interp.best_no_id_drop else None}) "
        f"pro_ood(alpha={interp.best_pro_ood.alpha if interp.best_pro_ood else None})"
    )


__all__ = [
    "ProbeConfig",
    "ProbeData",
    "ProbeTrainResult",
    "InterpolationPoint",
    "InterpolationResult",
    "ProbeExperimentResult",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_WARMUP_LR",
    "DEFAULT_WARMUP_TYPE",
    "DEFAULT_WARMUP_RATIO",
    "DEFAULT_LAMBDA_WEIGHT",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_ALIGNMENT_MODE",
    "DEFAULT_INTERP_GRID",
    "build_linear_warmup_cosine_scheduler",
    "build_optimizer",
    "set_seed",
    "make_feature_loader",
    "make_labeled_folds",
    "build_linear_probe",
    "get_probe_weights",
    "get_probe_bias",
    "set_probe_weights",
    "build_soft_loss",
    "build_hierarchy_soft_loss",
    "probe_logits",
    "probe_accuracy",
    "probe_top5_accuracy",
    "evaluate_probe",
    "train_linear_probe",
    "interpolate_probes",
    "interpolation_grid",
    "interpolate_and_evaluate",
    "run_linear_probe_experiment",
    "load_probe_data_from_cache",
]


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _self_test()
