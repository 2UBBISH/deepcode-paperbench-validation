"""Evaluation utilities for Sample-specific Multi-channel Masks (SMM).

This module implements the evaluation protocol described in the SMM paper
(ICML 2024):

* Section 5 "Baselines" reports *"the averaged test accuracy"* over **three
  seeds** on a single A100 GPU.  Top-1 accuracy is the reported metric.
* Section 5 "Feature Space Visualization Results" states that the t-SNE
  visualization uses the *"output layer feature before the label mapping
  layer"*.  The Addendum clarifies that *"the embeddings are computed using
  5000 randomly selected samples from each training set"*.

Concretely this file provides:

1. :func:`evaluate` - top-1 (and optional top-k) accuracy of a frozen
   pre-trained classifier fed with reprogrammed inputs, decoded through the
   output label mapping ``f_out`` (Ilm / Flm / Rlm).
2. :func:`extract_features` - the **pre-label-mapping** output-layer features
   ``f_P(f_in(x))`` used for the t-SNE analysis.
3. :func:`evaluate_seeds` - the three-seed mean +/- std reporting used by
   every table in the paper.
4. :func:`evaluate_method` - a convenience wrapper that trains a method
   (SMM or a shared-mask baseline) for each seed and reports mean +/- std.

All functions are non-learnable: no parameters live in this module.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import (
    AccuracyMeter,
    RunResult,
    aggregate_results,
    aggregate_seeds,
    compare_with_reference,
    dump_results_json,
    format_mean_std,
    format_results_table,
    mean_over_datasets,
    topk_accuracy,
)
from .seeds import SEEDS, dataloader_seed_kwargs, resolve_seeds, set_seed


__all__ = [
    "EvaluationConfig",
    "EvaluationResult",
    "decode_predictions",
    "evaluate",
    "evaluate_seeds",
    "evaluate_method",
    "extract_features",
    "extract_features_for_tsne",
    "collect_logits",
    "summarize_results",
    "evaluate_many_datasets",
    "EVAL_BATCH_SIZE",
    "TSNE_SAMPLES_PER_DATASET",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Batch size used for evaluation (mirrors the Table 9 training batch size).
EVAL_BATCH_SIZE = 256

#: Number of samples per training set used for the t-SNE analysis (Addendum).
TSNE_SAMPLES_PER_DATASET = 5000


# ---------------------------------------------------------------------------
# Configuration / result containers
# ---------------------------------------------------------------------------


@dataclass
class EvaluationConfig:
    """Evaluation-side knobs mirroring the training configuration.

    Attributes:
        backbone: name of the frozen pre-trained classifier (``resnet18``,
            ``resnet50`` or ``vit_b32``); used to derive the input resolution
            (224 for ResNets, 384 for ViT-B32) when ``input_size`` is unset.
        input_size: spatial size of the reprogrammed input ``f_in(x)``.
        batch_size: batch size used for the evaluation loader.
        num_workers: ``DataLoader`` worker count.
        device: torch device string; auto-detected when ``None``.
        max_batches: optional cap on the number of evaluated batches (debug).
        label_mapping: name of the output mapping (``ilm`` / ``flm`` / ``rlm``).
        topk: values of ``k`` for the auxiliary top-k metrics.
        save_predictions: when ``True``, prediction tensors are retained on the
            result object (needed for the t-SNE / confusion analyses).
        seeds: seeds to average over (paper: ``(0, 1, 2)``).
    """

    backbone: str = "resnet18"
    input_size: Optional[int] = None
    batch_size: int = EVAL_BATCH_SIZE
    num_workers: int = 4
    device: Optional[str] = None
    max_batches: Optional[int] = None
    label_mapping: str = "ilm"
    topk: Sequence[int] = (1, 5)
    save_predictions: bool = False
    seeds: Sequence[int] = SEEDS

    def as_dict(self) -> Dict[str, Any]:
        payload = dict(self.__dict__)
        payload["topk"] = list(self.topk)
        payload["seeds"] = list(self.seeds)
        return payload


@dataclass
class EvaluationResult:
    """Outcome of one evaluation pass (one dataset / method / seed)."""

    dataset: str = ""
    backbone: str = "resnet18"
    method: str = "ours"
    label_mapping: str = "ilm"
    seed: int = 0
    top1: float = 0.0
    top5: Optional[float] = None
    loss: Optional[float] = None
    num_samples: int = 0
    num_classes: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)
    predictions: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None

    @property
    def accuracy(self) -> float:
        """Top-1 accuracy in percent (paper convention)."""
        return self.top1

    def as_dict(self, include_predictions: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "dataset": self.dataset,
            "backbone": self.backbone,
            "method": self.method,
            "label_mapping": self.label_mapping,
            "seed": self.seed,
            "top1": self.top1,
            "top5": self.top5,
            "loss": self.loss,
            "num_samples": self.num_samples,
            "num_classes": self.num_classes,
        }
        if self.extra:
            payload["extra"] = self.extra
        if include_predictions:
            if self.predictions is not None:
                payload["predictions"] = self.predictions.tolist()
            if self.targets is not None:
                payload["targets"] = self.targets.tolist()
        return payload

    def as_run_result(self) -> RunResult:
        """Convert to the shared :class:`~smm_vr.engine.metrics.RunResult`."""
        return RunResult(
            dataset=self.dataset,
            backbone=self.backbone,
            method=self.method,
            seed=self.seed,
            accuracy=self.top1,
            loss=self.loss,
            mapping=self.label_mapping,
            extra=dict(self.extra),
        )

    def __str__(self) -> str:  # pragma: no cover - formatting helper
        return (
            f"[{self.dataset} | {self.backbone} | {self.method} | "
            f"{self.label_mapping} | seed={self.seed}] "
            f"top1={self.top1:.2f}%"
        )


# ---------------------------------------------------------------------------
# Label-mapping aware prediction decoding
# ---------------------------------------------------------------------------


def decode_predictions(
    logits: torch.Tensor,
    f_out: Optional[Any] = None,
) -> torch.Tensor:
    """Decode raw classifier logits into **target-space** predictions.

    SMM trains the reprogramming function ``f_in`` so that the frozen
    pre-trained classifier ``f_P`` predicts a mapped ImageNet label for each
    target class (Section 2.3).  Evaluation must therefore apply the same
    mapping ``f_out`` before computing accuracy.

    Three mapping conventions are supported (see
    ``smm_vr/label_mapping/flm.py`` and ``rlm.py``):

    * ``select`` (Flm / Ilm): ``target_to_pretrained`` is a
      ``(|Y^T|,)``-shaped index tensor; gathering those ImageNet columns
      yields ``(|Y^T|,)`` logits whose argmax is directly the target class
      index.
    * ``inject`` (Rlm): logits live in the full ImageNet space and
      ``pretrained_to_target[p]`` maps each ImageNet label to its target
      class; entries equal to ``IGNORE_INDEX`` are excluded.
    * ``None``: identity - the model already outputs target-space logits.

    Args:
        logits: ``(B, |Y^P|)`` or ``(B, |Y^T|)`` float tensor.
        f_out: optional output label mapping object or plain index tensor
            with shape ``(|Y^T|,)`` describing ``t -> y^P``.

    Returns:
        ``(B,)`` long tensor of predicted target-class indices.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    if f_out is None:
        return logits.argmax(dim=1)

    # Bare index tensor describing t -> y^P (an Flm/Ilm-style selection).
    if torch.is_tensor(f_out) and not isinstance(f_out, nn.Module):
        index = f_out.to(logits.device).long()
        valid = index >= 0
        if not bool(valid.all()):
            # Unmatched ImageNet labels produce -inf so they are never
            # selected; their target class is unreachable, matching Ilm/Flm.
            gathered = torch.full(
                (logits.shape[0], index.numel()),
                float("-inf"),
                device=logits.device,
                dtype=logits.dtype,
            )
            if bool(valid.any()):
                gathered[:, valid] = logits[:, index[valid]]
            return gathered.argmax(dim=1)
        return logits[:, index].argmax(dim=1)

    # Module-style mapping: ask it directly (all mapping classes implement
    # ``forward`` on logits, returning mapped logits).
    if callable(f_out):
        try:
            mapped = f_out(logits)
        except TypeError:
            mapped = f_out(logits, logits.device)
        if torch.is_tensor(mapped) and mapped.dim() == 2 and mapped.shape[0] == logits.shape[0]:
            return _argmax_ignoring_sentinels(mapped, f_out)
        if torch.is_tensor(mapped):
            return mapped.long().view(-1)

    raise TypeError(
        "f_out must be None, an index tensor, or a callable mapping logits "
        f"to target-space logits; got {type(f_out)!r}."
    )


def _argmax_ignoring_sentinels(logits: torch.Tensor, f_out: Any) -> torch.Tensor:
    """Argmax over mapped logits, treating ``IGNORE_INDEX`` rows as invalid."""
    ignore_index = getattr(f_out, "ignore_index", -1)
    if ignore_index is None or ignore_index < 0:
        return logits.argmax(dim=1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    if 0 <= ignore_index < logits.shape[1]:
        mask[:, ignore_index] = False
    if not bool(mask.any()):
        return logits.argmax(dim=1)
    filled = logits.masked_fill(~mask, float("-inf"))
    return filled.argmax(dim=1)


# ---------------------------------------------------------------------------
# Core evaluation pass
# ---------------------------------------------------------------------------


def _resolve_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """Split a dataloader batch into ``(images, targets, extra)``."""
    if isinstance(batch, (list, tuple)):
        images = batch[0]
        targets = batch[1] if len(batch) > 1 else None
        extra = batch[2] if len(batch) > 2 else None
        return images, targets, extra
    return batch, None, None


@torch.no_grad()
def collect_logits(
    classifier: nn.Module,
    data_loader: Iterable[Any],
    *,
    f_in: Optional[nn.Module] = None,
    device: Optional[str] = None,
    max_batches: Optional[int] = None,
    return_targets: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Collect the frozen classifier's logits for every evaluated sample.

    Args:
        classifier: frozen pre-trained model ``f_P`` (ImageNet-1K logits).
        data_loader: iterable yielding ``(images, targets)`` batches.
        f_in: optional reprogramming wrapper applied before ``f_P``.
        device: evaluation device.
        max_batches: optional cap for debugging runs.
        return_targets: whether to also stack the targets.

    Returns:
        Tuple ``(logits, targets)`` with ``logits`` of shape
        ``(N, |Y^P|)`` and ``targets`` of shape ``(N,)`` (or ``None``).
    """
    device = _resolve_device(device)
    if f_in is not None:
        f_in.eval().to(device)
    classifier.eval().to(device)

    logit_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []

    for batch_idx, batch in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, targets, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        if f_in is not None:
            images = f_in(images)
        logits = classifier(images)
        logit_chunks.append(logits.detach().float().cpu())
        if return_targets and targets is not None:
            target_chunks.append(targets.detach().cpu().long())

    if not logit_chunks:
        empty = torch.empty(0, 0, dtype=torch.float32)
        return empty, (torch.empty(0, dtype=torch.long) if return_targets else None)

    logits_all = torch.cat(logit_chunks, dim=0)
    targets_all: Optional[torch.Tensor] = None
    if return_targets:
        if target_chunks:
            targets_all = torch.cat(target_chunks, dim=0)
        else:
            targets_all = torch.empty(0, dtype=torch.long)
    return logits_all, targets_all


def evaluate(
    classifier: nn.Module,
    data_loader: Iterable[Any],
    *,
    f_in: Optional[nn.Module] = None,
    f_out: Optional[Any] = None,
    device: Optional[str] = None,
    max_batches: Optional[int] = None,
    topk: Sequence[int] = (1, 5),
    dataset: str = "",
    method: str = "ours",
    label_mapping: str = "ilm",
    backbone: str = "resnet18",
    seed: int = 0,
    num_classes: Optional[int] = None,
    keep_predictions: bool = False,
    criterion: Optional[nn.Module] = None,
) -> EvaluationResult:
    """Evaluate a (possibly reprogrammed) classifier's top-1 test accuracy.

    Implements the paper's evaluation protocol: sample-specific masks are
    applied to the frozen pre-trained classifier via ``f_in``, the classifier's
    output-layer logits are decoded through the output mapping ``f_out``, and
    top-1 accuracy is averaged over the test set (reported in percent).

    Args:
        classifier: frozen pre-trained classifier ``f_P``.
        data_loader: test (or train) loader.
        f_in: the SMM / baseline reprogramming wrapper; ``None`` evaluates the
            unmodified image pipeline.
        f_out: output label mapping (Ilm/Flm/Rlm object, index tensor, or
            ``None`` for identity).
        device: torch device.
        max_batches: debug cap on batches.
        topk: ``k`` values for the auxiliary top-k metrics.
        dataset: dataset name, stored on the result.
        method: method name (``ours``, ``pad``, ``narrow``, ``medium``,
            ``full``, ``only_delta``, ``only_fmask``, ``single_channel`` ...).
        label_mapping: mapping name, stored on the result.
        backbone: backbone name, stored on the result.
        seed: seed of this run, stored on the result.
        num_classes: number of target classes (inferred from targets if unset).
        keep_predictions: retain predictions/targets on the result object.
        criterion: optional loss module for reporting the evaluation loss.

    Returns:
        :class:`EvaluationResult` with ``top1`` in percent.
    """
    logits, targets = collect_logits(
        classifier,
        data_loader,
        f_in=f_in,
        device=device,
        max_batches=max_batches,
        return_targets=True,
    )

    result = EvaluationResult(
        dataset=dataset,
        backbone=backbone,
        method=method,
        label_mapping=label_mapping,
        seed=seed,
        num_samples=int(logits.shape[0]) if logits.numel() else 0,
        num_classes=int(num_classes or 0),
    )

    if logits.numel() == 0 or targets is None or targets.numel() == 0:
        return result

    predicted = decode_predictions(logits, f_out)
    targets = targets.long()

    if num_classes is None and targets.numel():
        result.num_classes = int(targets.max().item()) + 1

    # Top-1 / top-k through the shared accuracy meter (percent units).
    meter = AccuracyMeter(topk=tuple(sorted(set(topk) | {1})))
    meter.update(predicted, targets)
    result.top1 = meter.get(1)
    if 5 in meter.topk_values:
        result.top5 = meter.get(5)

    if criterion is not None:
        with torch.no_grad():
            mapped = _mapped_logits_for_loss(logits, f_out)
            if mapped is not None and mapped.shape[1] == result.num_classes:
                result.loss = float(criterion(mapped, targets).item())

    if keep_predictions:
        result.predictions = predicted.detach().cpu()
        result.targets = targets.detach().cpu()

    return result


def _mapped_logits_for_loss(logits: torch.Tensor, f_out: Optional[Any]) -> Optional[torch.Tensor]:
    """Return the target-space logit matrix used by cross-entropy."""
    if f_out is None:
        return logits
    if torch.is_tensor(f_out) and not isinstance(f_out, nn.Module):
        index = f_out.long()
        if bool((index < 0).any()):
            return None
        return logits[:, index]
    if callable(f_out):
        try:
            mapped = f_out(logits)
        except TypeError:
            return None
        if torch.is_tensor(mapped) and mapped.dim() == 2:
            return mapped
    return None


# ---------------------------------------------------------------------------
# Feature extraction (pre-label-mapping output-layer features)
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_features(
    classifier: nn.Module,
    data_loader: Iterable[Any],
    *,
    f_in: Optional[nn.Module] = None,
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
    max_batches: Optional[int] = None,
    normalize: bool = False,
    return_targets: bool = True,
    seed: Optional[int] = None,
    module: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extract the output-layer features **before the label mapping layer**.

    Section 5 ("Feature Space Visualization Results") visualizes *"the output
    layer feature before the label mapping layer"*.  Because the SMM label
    mapping is non-parametric and simply selects ImageNet logits, these
    features are exactly the penultimate feature vector of ``f_P`` evaluated on
    the reprogrammed input ``f_in(x)``.

    the classifier is run up to the penultimate layer; if the penultimate
    module cannot be identified automatically, the raw logit vectors are
    returned instead (still "before the label mapping layer").

    Args:
        classifier: frozen pre-trained classifier ``f_P``.
        data_loader: loader yielding ``(images, targets)`` batches.
        f_in: optional reprogramming wrapper applied before ``f_P``.
        device: torch device.
        max_samples: cap on the number of extracted feature vectors (the
            Addendum asks for 5000 randomly selected training samples per
            dataset).
        max_batches: alternative cap expressed in batches.
        normalize: if ``True``, L2-normalise each feature vector.
        return_targets: whether to return the class labels as well.
        seed: when ``max_samples`` is used, the loader is expected to be a
            shuffled (randomly selected) subset; the seed is only recorded for
            reproducibility of downstream analysis.
        module: optional explicit feature module (e.g. an intermediate layer).

    Returns:
        ``(features, targets)`` where ``features`` has shape
        ``(N, D)`` and ``targets`` (if requested) shape ``(N,)``.
    """
    device = _resolve_device(device)
    if f_in is not None:
        f_in.eval().to(device)
    classifier.eval().to(device)

    feature_fn = _build_feature_extractor(classifier, module=module)

    feature_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []
    collected = 0

    for batch_idx, batch in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if max_samples is not None and collected >= max_samples:
            break
        images, targets, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        if f_in is not None:
            images = f_in(images)
        feats = feature_fn(images)
        feats = feats.detach().float().cpu()
        if feats.dim() > 2:
            feats = feats.flatten(start_dim=1)
        if return_targets and targets is not None:
            tgt = targets.detach().cpu().long()
            target_chunks.append(tgt)
        feature_chunks.append(feats)
        collected += int(feats.shape[0])

    if not feature_chunks:
        return torch.empty(0, 0, dtype=torch.float32), (
            torch.empty(0, dtype=torch.long) if return_targets else None
        )

    features = torch.cat(feature_chunks, dim=0)
    targets_all: Optional[torch.Tensor] = None
    if return_targets:
        targets_all = (
            torch.cat(target_chunks, dim=0) if target_chunks else torch.empty(0, dtype=torch.long)
        )

    if max_samples is not None and features.shape[0] > max_samples:
        features = features[:max_samples]
        if targets_all is not None and targets_all.numel():
            targets_all = targets_all[:max_samples]

    if normalize and features.numel():
        features = F.normalize(features, dim=1)

    return features, targets_all


def _build_feature_extractor(
    classifier: nn.Module,
    module: Optional[nn.Module] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a callable mapping images to penultimate ("output-layer") features."""
    if module is not None:
        return _make_forward_hook_fn(classifier, module)

    for attr in ("avgpool", "global_pool", "pool", "features"):
        candidate = getattr(classifier, attr, None)
        if isinstance(candidate, nn.Module):
            return _make_forward_hook_fn(classifier, candidate)
    # Fall back to the raw logits ("before the label mapping layer").
    return lambda images: classifier(images)


def _make_forward_hook_fn(
    classifier: nn.Module,
    target_module: nn.Module,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a function that runs ``classifier`` and returns ``target_module``'s output."""
    captured: Dict[str, torch.Tensor] = {}

    def _hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
        captured["value"] = output.detach()

    handle = target_module.register_forward_hook(_hook)

    def _fn(images: torch.Tensor) -> torch.Tensor:
        captured.pop("value", None)
        classifier(images)
        value = captured.get("value")
        if value is None:  # pragma: no cover - defensive
            raise RuntimeError(
                "Feature extraction hook did not capture an output; the target "
                "module may not be part of the classifier's forward pass."
            )
        return value

    # Attach the handle to the returned function so callers can remove it.
    _fn.__dict__["_hook_handle"] = handle  # type: ignore[attr-defined]
    return _fn


def extract_features_for_tsne(
    classifier: nn.Module,
    train_dataset: Any,
    *,
    f_in: Optional[nn.Module] = None,
    num_samples: int = TSNE_SAMPLES_PER_DATASET,
    batch_size: int = EVAL_BATCH_SIZE,
    num_workers: int = 4,
    device: Optional[str] = None,
    seed: int = 0,
    backbone: str = "resnet18",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract the features used by the t-SNE figures.

    The Addendum specifies that *"the embeddings are computed using 5000
    randomly selected samples from each training set"*.  A deterministic
    permutation of the training set (driven by ``seed``) is used to draw the
    ``num_samples`` samples, exactly as described.

    Args:
        classifier: frozen pre-trained classifier ``f_P``.
        train_dataset: the training split (with the paper's train transform).
        f_in: reprogramming wrapper (``None`` = no reprogramming baseline).
        num_samples: number of randomly selected training samples (5000).
        batch_size: evaluation batch size.
        num_workers: number of dataloader workers.
        device: torch device.
        seed: seed of the random sample selection.
        backbone: backbone name, used for logging only.

    Returns:
        ``(features, targets)`` with ``features`` of shape
        ``(min(num_samples, len(train_dataset)), D)``.
    """
    from torch.utils.data import DataLoader, Subset

    total = len(train_dataset)
    n = min(int(num_samples), int(total))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(total, generator=generator)[:n]
    subset = Subset(train_dataset, perm.tolist())

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **dataloader_seed_kwargs(seed),
    )
    return extract_features(
        classifier,
        loader,
        f_in=f_in,
        device=device,
        max_samples=n,
        normalize=False,
        return_targets=True,
    )


# ---------------------------------------------------------------------------
# Three-seed reporting (Section 5 "Baselines")
# ---------------------------------------------------------------------------


def evaluate_seeds(
    classifier: nn.Module,
    data_loaders: Sequence[Iterable[Any]],
    *,
    f_in_factory: Optional[Callable[[int], Optional[nn.Module]]] = None,
    f_out_factory: Optional[Callable[[int], Optional[Any]]] = None,
    seeds: Optional[Sequence[int]] = None,
    device: Optional[str] = None,
    topk: Sequence[int] = (1, 5),
    dataset: str = "",
    method: str = "ours",
    label_mapping: str = "ilm",
    backbone: str = "resnet18",
    num_classes: Optional[int] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate one method over the paper's three seeds and aggregate.

    "Experiments are run with three seeds on a single A100 GPU and the
    averaged test accuracy is reported." (Section 5, *Baselines*)

    Args:
        classifier: frozen pre-trained classifier (or a factory-callable
            accepting a seed when per-seed reloading is desired).
        data_loaders: one evaluation loader per seed, or a single loader.
        f_in_factory: callable ``seed -> f_in`` supplying the reprogramming
            wrapper for each seed (``None`` = no reprogramming).
        f_out_factory: callable ``seed -> f_out`` supplying the mapping.
        seeds: seeds to average (default ``(0, 1, 2)``).
        device: torch device.
        topk: top-k metrics to compute.
        dataset: dataset name.
        method: method name.
        label_mapping: mapping name.
        backbone: backbone name.
        num_classes: target class count.
        max_batches: debug batch cap.

    Returns:
        Dictionary with ``per_seed_accuracy``, ``mean``, ``std``,
        ``formatted``, ``results`` (list of :class:`EvaluationResult`) and
        ``result`` (an aggregated :class:`RunResult`).
    """
    seed_list = resolve_seeds(seeds)
    if len(data_loaders) == 1:
        loaders = [data_loaders[0]] * len(seed_list)
    else:
        if len(data_loaders) != len(seed_list):
            raise ValueError(
                f"Expected 1 or {len(seed_list)} evaluation loaders, "
                f"got {len(data_loaders)}."
            )
        loaders = list(data_loaders)

    results: List[EvaluationResult] = []
    for seed, loader in zip(seed_list, loaders):
        set_seed(seed)
        f_in = f_in_factory(seed) if f_in_factory is not None else None
        f_out = f_out_factory(seed) if f_out_factory is not None else None
        model = classifier(seed) if callable(classifier) and not isinstance(classifier, nn.Module) else classifier
        results.append(
            evaluate(
                model,
                loader,
                f_in=f_in,
                f_out=f_out,
                device=device,
                max_batches=max_batches,
                topk=topk,
                dataset=dataset,
                method=method,
                label_mapping=label_mapping,
                backbone=backbone,
                seed=seed,
                num_classes=num_classes,
            )
        )

    accuracies = [r.top1 for r in results]
    mean, std = aggregate_seeds(accuracies)
    return {
        "dataset": dataset,
        "backbone": backbone,
        "method": method,
        "label_mapping": label_mapping,
        "seeds": list(seed_list),
        "per_seed_accuracy": accuracies,
        "mean": mean,
        "std": std,
        "formatted": format_mean_std(mean, std),
        "results": results,
        "result": RunResult(
            dataset=dataset,
            backbone=backbone,
            method=method,
            seed=-1,
            accuracy=mean,
            loss=None,
            mapping=label_mapping,
            extra={"std": std, "per_seed_accuracy": accuracies},
        ),
    }


def evaluate_method(
    train_fn: Callable[..., Any],
    *,
    dataset: str,
    method: str = "ours",
    backbone: str = "resnet18",
    label_mapping: str = "ilm",
    seeds: Optional[Sequence[int]] = None,
    device: Optional[str] = None,
    num_classes: Optional[int] = None,
    **train_kwargs: Any,
) -> Dict[str, Any]:
    """Train + evaluate a method for every seed and return mean +/- std.

    ``train_fn`` is expected to return either a full training bundle
    (``dict`` with a ``history``/``result`` entry) or a
    :class:`~smm_vr.engine.metrics.RunResult` / accuracy value per seed.

    Args:
        train_fn: callable ``(seed=..., **kwargs) -> result``.
        dataset: dataset name.
        method: method name reported in the tables.
        backbone: backbone name.
        label_mapping: output mapping name.
        seeds: seeds to run (default ``(0, 1, 2)``).
        device: torch device.
        num_classes: target class count.
        **train_kwargs: forwarded to ``train_fn``.

    Returns:
        Aggregated dictionary (same schema as :func:`evaluate_seeds`).
    """
    seed_list = resolve_seeds(seeds)
    per_seed: List[float] = []
    extras: List[Dict[str, Any]] = []

    for seed in seed_list:
        set_seed(seed)
        out = train_fn(seed=seed, device=device, **train_kwargs)
        accuracy = _extract_accuracy(out)
        per_seed.append(float(accuracy))
        extras.append(out if isinstance(out, dict) else {})

    mean, std = aggregate_seeds(per_seed)
    return {
        "dataset": dataset,
        "backbone": backbone,
        "method": method,
        "label_mapping": label_mapping,
        "seeds": list(seed_list),
        "per_seed_accuracy": per_seed,
        "mean": mean,
        "std": std,
        "formatted": format_mean_std(mean, std),
        "num_classes": num_classes,
        "runs": extras,
        "result": RunResult(
            dataset=dataset,
            backbone=backbone,
            method=method,
            seed=-1,
            accuracy=mean,
            loss=None,
            mapping=label_mapping,
            extra={"std": std, "per_seed_accuracy": per_seed},
        ),
    }


def _extract_accuracy(out: Any) -> float:
    """Pull a scalar accuracy out of the many possible training return types."""
    if isinstance(out, (int, float)):
        return float(out)
    if hasattr(out, "accuracy"):
        return float(getattr(out, "accuracy"))
    if hasattr(out, "final_test_accuracy"):
        value = getattr(out, "final_test_accuracy")
        return float(value) if value is not None else float("nan")
    if isinstance(out, dict):
        for key in ("mean", "accuracy", "final_test_accuracy", "best_test_accuracy", "top1"):
            if key in out and out[key] is not None:
                value = out[key]
                return float(value)
        for key in ("result", "history"):
            if key in out:
                try:
                    return _extract_accuracy(out[key])
                except (TypeError, ValueError):
                    continue
    raise TypeError(f"Unable to extract an accuracy from {type(out)!r}.")


# ---------------------------------------------------------------------------
# Multi-dataset reporting helpers
# ---------------------------------------------------------------------------


def summarize_results(
    results: Sequence[Any],
    *,
    dataset_order: Optional[Sequence[str]] = None,
    method: str = "ours",
    decimals: int = 1,
) -> Dict[str, Any]:
    """Build per-dataset means plus the paper's highlighted ``AVERAGE`` row.

    Args:
        results: :class:`EvaluationResult` objects, :class:`RunResult` objects
            or plain dictionaries with ``dataset`` and ``accuracy``/``top1``.
        dataset_order: order of the datasets in the rendered table.
        method: which method to keep when several methods are present.
        decimals: decimal places for the rendered strings.

    Returns:
        ``{"per_dataset": {name: mean}, "average": mean, "table": str}``.
    """
    by_dataset: Dict[str, List[float]] = {}
    for item in results:
        name, value, item_method = _unpack_summary_item(item)
        if item_method is not None and method is not None and item_method != method:
            continue
        by_dataset.setdefault(name, []).append(value)

    per_dataset = {name: sum(vals) / len(vals) for name, vals in by_dataset.items()}
    if dataset_order is not None:
        ordered = {name: per_dataset[name] for name in dataset_order if name in per_dataset}
        for name, value in per_dataset.items():
            ordered.setdefault(name, value)
        per_dataset = ordered

    average = mean_over_datasets(per_dataset)
    table = format_results_table(per_dataset, dataset_order=dataset_order,
                                 method=method, decimals=decimals)
    return {"per_dataset": per_dataset, "average": average, "table": table}


def _unpack_summary_item(item: Any) -> Tuple[str, float, Optional[str]]:
    if isinstance(item, dict):
        name = str(item.get("dataset", ""))
        value = item.get("accuracy", item.get("top1", item.get("mean")))
        method = item.get("method")
        return name, float(value), method
    if isinstance(item, RunResult):
        return item.dataset, float(item.accuracy), item.method
    if isinstance(item, EvaluationResult):
        return item.dataset, float(item.top1), item.method
    if hasattr(item, "dataset") and (hasattr(item, "top1") or hasattr(item, "accuracy")):
        value = getattr(item, "top1", getattr(item, "accuracy"))
        return str(getattr(item, "dataset")), float(value), getattr(item, "method", None)
    raise TypeError(f"Unsupported summary item type: {type(item)!r}")


def evaluate_many_datasets(
    datasets: Sequence[str],
    runner: Callable[[str], Dict[str, Any]],
    *,
    method: str = "ours",
    dataset_order: Optional[Sequence[str]] = None,
    reference_table: Optional[str] = None,
    decimals: int = 1,
) -> Dict[str, Any]:
    """Evaluate SMM on many datasets and summarize like the paper's tables.

    Args:
        datasets: dataset names to evaluate.
        runner: callable ``dataset -> aggregated dict`` (e.g.
            :func:`evaluate_method` output).
        method: method name for the summary filter.
        dataset_order: rendering order (defaults to ``datasets``).
        reference_table: optional name of a table in
            :mod:`smm_vr.data.dataset_stats` / :mod:`smm_vr.engine.metrics` to
            compare against (``table1_resnet18``, ``table2_vit_b32``, ...).
        decimals: decimals used when rendering the table.

    Returns:
        Summary dict from :func:`summarize_results` extended with
        ``runs`` (per dataset aggregates) and ``comparison`` (when a reference
        table is supplied).
    """
    runs: Dict[str, Dict[str, Any]] = {}
    rows: List[RunResult] = []
    for name in datasets:
        out = runner(name)
        runs[name] = out
        if isinstance(out, dict) and "result" in out:
            rows.append(out["result"])
        elif isinstance(out, dict):
            rows.append(
                RunResult(
                    dataset=name,
                    method=method,
                    accuracy=float(out.get("mean", out.get("accuracy", float("nan")))),
                )
            )
        else:
            rows.append(RunResult(dataset=name, method=method, accuracy=_extract_accuracy(out)))

    summary = summarize_results(
        rows,
        dataset_order=dataset_order or list(datasets),
        method=method,
        decimals=decimals,
    )
    summary["runs"] = runs

    if reference_table is not None:
        summary["comparison"] = compare_with_reference(
            summary["per_dataset"],
            table=reference_table,
            key=method,
        )
    return summary


def save_results(path: str, results: Sequence[Any], **metadata: Any) -> str:
    """Persist evaluation results (and metadata) to JSON."""
    rows = [
        r.as_dict() if hasattr(r, "as_dict") else dict(r) for r in results
    ]
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    return dump_results_json(path, rows, **metadata)
