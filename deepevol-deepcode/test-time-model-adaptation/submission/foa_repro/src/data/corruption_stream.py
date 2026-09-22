"""Non-i.i.d. test streams for FOA (paper Section 4.4 / Table 11).

The paper verifies the effectiveness of FOA under two non-i.i.d. scenarios by
following NOTE (Gong et al., 2022) and SAR (Niu et al., 2023):

1. **online imbalanced label distribution shifts** -- "test data come in a class
   order", i.e. the online stream is sorted by the ground-truth class label so
   that the label distribution seen so far is heavily imbalanced.  For this
   scenario the paper reports the *average* result over the 15 corruptions
   (Table 11: FOA 62.1% acc / 6.6% ECE, TENT 60.2 / 17.7, SAR 60.8 / 7.5).

2. **mixed domain shifts** -- "test data stream consists of multiple randomly
   mixed domains with different distribution shifts", i.e. a *single* stream in
   which the 15 ImageNet-C corruptions are concatenated in a random order.
   Table 11: FOA 62.0% acc / 4.9% ECE, TENT 56.9 / 29.2, SAR 61.4 / 14.8.

Both scenarios are single-pass (no shuffling of the stream once its global order
is fixed) exactly like the mild/i.i.d. ImageNet-C protocol used elsewhere in the
code base.  The batches yielded by :class:`NonIIDStream` use the same dictionary
layout as :class:`src.data.datasets.OnlineStream` so that the FOA drivers can be
reused without modification:

    {"image": [...], "label": [...], "domain": [...],  # domain only for mixed
     "images": ...,  "x": ..., "labels": ..., "y": ...}

Nothing in this module touches the model: it is pure data plumbing.
"""

from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from .datasets import (
    IMAGENET_C_CORRUPTIONS,
    DATASET_NUM_CLASSES,
    OnlineStream,
    build_dataset,
    build_transform,
    normalize_dataset_name,
)

__all__ = [
    "NON_IID_SCENARIOS",
    "LABEL_SHIFT_MODES",
    "class_ordered_indices",
    "label_shift_indices",
    "MixedDomainDataset",
    "NonIIDStream",
    "build_non_iid_stream",
    "build_label_shift_stream",
    "build_mixed_domain_stream",
    "non_iid_summary_key",
]

#: The three stream regimes reported in Table 11.
NON_IID_SCENARIOS = ("iid", "label_shift", "mixed_shift")

#: Supported imbalance profiles for the class-ordered ("online imbalanced label
#: distribution shift") stream.
#:   * ``"class_order"``  : pure class ordering, each class equally represented
#:                          (the default reading of "test data come in a class
#:                          order" in Section 4.4).
#:   * ``"geometric"``    : additional geometric decay of the per-class sample
#:                          budget, i.e. class ``k`` keeps ``decay ** k`` of its
#:                          samples -> an explicitly *imbalanced* label stream.
#:   * ``"dirichlet"``    : per-class sample counts drawn from a symmetric
#:                          Dirichlet distribution (NOTA-style imbalance).
LABEL_SHIFT_MODES = ("class_order", "geometric", "dirichlet")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Read a dotted config key from a dict-like / attribute-like config."""
    if cfg is None:
        return default
    cur = cfg
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            if key in cur:
                cur = cur[key]
            else:
                return default
        else:  # attribute access (src.utils.config.Config)
            try:
                cur = getattr(cur, key)
            except (AttributeError, KeyError):
                try:
                    cur = cur[key]  # dict subclass fallback
                except Exception:
                    return default
    return cur


def _dataset_labels(dataset: Dataset) -> np.ndarray:
    """Best-effort extraction of the label of every sample of ``dataset``."""
    for attr in ("targets", "labels", "ys"):
        val = getattr(dataset, attr, None)
        if val is not None and not callable(val):
            return np.asarray(val, dtype=np.int64)
    targets_fn = getattr(dataset, "targets", None)
    if callable(targets_fn):
        return np.asarray(targets_fn(), dtype=np.int64)
    if isinstance(dataset, Subset):
        base = _dataset_labels(dataset.dataset)
        return base[np.asarray(dataset.indices, dtype=np.int64)]
    if isinstance(dataset, ConcatDataset):
        parts = [_dataset_labels(d) for d in dataset.datasets]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    # Slow path: query each sample (only used for exotic dataset wrappers).
    labels: List[int] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if isinstance(sample, dict):
            lab = sample.get("label", sample.get("labels", sample.get("y")))
        elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
            lab = sample[1]
        else:  # pragma: no cover - dataset without labels
            raise ValueError("Unable to determine labels of dataset %r" % (type(dataset),))
        if torch.is_tensor(lab):
            lab = lab.item()
        labels.append(int(lab))
    return np.asarray(labels, dtype=np.int64)


def _dataset_domain_labels(dataset: Dataset) -> Optional[np.ndarray]:
    """Return per-sample corruption-domain ids if the dataset exposes them."""
    for attr in ("domains", "domain_labels", "corruption_ids"):
        val = getattr(dataset, attr, None)
        if val is not None:
            return np.asarray(val, dtype=np.int64)
    return None


def class_ordered_indices(
    labels: Sequence[int],
    seed: int = 0,
    shuffle_within_class: bool = True,
) -> np.ndarray:
    """Order sample indices by ascending class label.

    Implements the "test data come in a class order" protocol of Section 4.4:
    every sample of class 0 is presented first, then every sample of class 1,
    and so on.  Samples inside a class may be shuffled (the *within-class* order
    is irrelevant to the label distribution but keeping it fixed/deterministic
    is required for reproducibility).

    Args:
        labels: ground-truth label of every sample of the dataset.
        seed: seed for the within-class permutation.
        shuffle_within_class: if ``False`` keep the original dataset order
            inside each class (fully deterministic stream).

    Returns:
        ``np.ndarray`` of dataset indices of length ``len(labels)``.
    """
    labels = np.asarray(labels, dtype=np.int64)
    rng = random.Random(seed)
    order: List[int] = []
    for cls in np.unique(labels):
        idx = np.nonzero(labels == cls)[0].tolist()
        if shuffle_within_class:
            rng.shuffle(idx)
        order.extend(idx)
    return np.asarray(order, dtype=np.int64)


def label_shift_indices(
    labels: Sequence[int],
    seed: int = 0,
    mode: str = "class_order",
    decay: float = 0.98,
    dirichlet_alpha: float = 0.5,
    shuffle_within_class: bool = True,
    min_per_class: int = 1,
) -> np.ndarray:
    """Class-ordered indices with an optional, explicitly imbalanced class budget.

    ``mode="class_order"`` reproduces the plain class-ordered stream.  The two
    other modes add the *imbalanced* label distribution described in Section 4.4
    (following NOTE / SAR): with ``"geometric"`` the number of retained samples
    decays geometrically with the class index, with ``"dirichlet"`` the per-class
    counts are sampled from ``Dirichlet(alpha * 1_C)``.
    """
    if mode not in LABEL_SHIFT_MODES:
        raise ValueError(
            "unknown label-shift mode %r (expected one of %s)" % (mode, LABEL_SHIFT_MODES)
        )
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels)
    rng = random.Random(seed)
    if mode == "class_order":
        return class_ordered_indices(labels, seed=seed, shuffle_within_class=shuffle_within_class)

    per_class_idx: Dict[int, List[int]] = {}
    for cls in classes:
        idx = np.nonzero(labels == cls)[0].tolist()
        if shuffle_within_class:
            rng.shuffle(idx)
        per_class_idx[int(cls)] = idx

    if mode == "geometric":
        weights = np.asarray([decay ** k for k in range(len(classes))], dtype=np.float64)
    else:  # dirichlet
        py_rng = np.random.RandomState(seed)
        weights = py_rng.dirichlet(np.full(len(classes), float(dirichlet_alpha)))
    weights = weights / max(weights.sum(), 1e-12)

    order: List[int] = []
    for k, cls in enumerate(classes):
        idx = per_class_idx[int(cls)]
        keep = int(round(weights[k] * len(labels)))
        keep = max(int(min_per_class), min(keep, len(idx)))
        order.extend(idx[:keep])
    return np.asarray(order, dtype=np.int64)


# ---------------------------------------------------------------------------
# datasets / streams
# ---------------------------------------------------------------------------
class MixedDomainDataset(Dataset):
    """Concatenation of several datasets in a (random) domain order.

    Used for the "mixed domain shifts" scenario: "test data stream consists of
    multiple randomly mixed domains with different distribution shifts".  The
    domains (e.g. the 15 ImageNet-C corruptions) are concatenated block-wise in a
    randomly permuted order and every sample carries the id of the domain it came
    from, which is handy for per-domain breakdowns.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        domain_names: Optional[Sequence[str]] = None,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if not datasets:
            raise ValueError("MixedDomainDataset requires at least one dataset")
        self.datasets = list(datasets)
        self.domain_names = list(domain_names) if domain_names is not None else [
            "domain_%d" % i for i in range(len(self.datasets))
        ]
        n_domains = len(self.datasets)
        self.domain_order = list(range(n_domains))
        if shuffle:
            random.Random(seed).shuffle(self.domain_order)
        self._order = [self.datasets[i] for i in self.domain_order]
        self._names = [self.domain_names[i] for i in self.domain_order]

        self._lengths = [len(d) for d in self._order]
        self._offsets = np.cumsum([0] + self._lengths).tolist()
        # Per-sample domain id (aligned with __getitem__ order).
        self.domains = np.concatenate(
            [
                np.full(n, i, dtype=np.int64)
                for i, n in enumerate(self._lengths)
                if n > 0
            ]
        ) if sum(self._lengths) > 0 else np.zeros(0, dtype=np.int64)

    # -- Dataset API ------------------------------------------------------
    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        for domain_id, (start, end) in enumerate(zip(self._offsets[:-1], self._offsets[1:])):
            if start <= index < end:
                return self._order[domain_id][index - start]
        raise IndexError(index)

    # -- helpers ----------------------------------------------------------
    def domain_of(self, index: int) -> int:
        return int(self.domains[index])

    def targets(self) -> List[int]:
        out: List[int] = []
        for ds in self._order:
            out.extend([int(t) for t in _dataset_labels(ds).tolist()])
        return out

    def domain_sizes(self) -> Dict[str, int]:
        return {name: int(n) for name, n in zip(self._names, self._lengths)}


class NonIIDStream:
    """Ordered, single-pass batched iterator over a non-i.i.d. stream.

    Mirrors :class:`src.data.datasets.OnlineStream`: batches are dictionaries with
    an ``"image"``/``"label"`` pair (plus the legacy aliases ``images``/``x`` and
    ``labels``/``y``) and, when available, a ``"domain"`` entry describing which
    corruption the sample came from.  The underlying order is fixed at
    construction time and never shuffled during iteration.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        num_workers: int = 4,
        device: Optional[torch.device] = None,
        drop_last: bool = False,
        dataset_name: str = "unknown",
        scenario: str = "label_shift",
        class_subset: Optional[Sequence[int]] = None,
        num_classes_eval: Optional[int] = None,
        collate_fn=None,
        pin_memory: bool = False,
        domains: Optional[Sequence[int]] = None,
        domain_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.device = device
        self.drop_last = bool(drop_last)
        self.dataset_name = dataset_name
        self.scenario = scenario
        self.class_subset = list(class_subset) if class_subset is not None else None
        if num_classes_eval is None:
            num_classes_eval = DATASET_NUM_CLASSES.get(normalize_dataset_name(dataset_name))
        self.num_classes_eval = num_classes_eval
        self.collate_fn = collate_fn
        self.pin_memory = bool(pin_memory)
        self.domains = np.asarray(domains, dtype=np.int64) if domains is not None else None
        self.domain_names = list(domain_names) if domain_names is not None else None

    # -- batch formatting --------------------------------------------------
    def _to_batch(self, images: torch.Tensor, labels: torch.Tensor, start: int) -> Dict[str, Any]:
        batch: Dict[str, Any] = {"image": images, "label": labels}
        batch["images"] = images
        batch["x"] = images
        batch["labels"] = labels
        batch["y"] = labels
        if self.domains is not None and len(self.domains):
            dom = self.domains[start : start + len(labels)]
            batch["domain"] = torch.as_tensor(dom, dtype=torch.long)
        if self.device is not None:
            batch["image"] = batch["image"].to(self.device)
            batch["images"] = batch["image"]
            batch["x"] = batch["image"]
            batch["label"] = batch["label"].to(self.device)
            batch["labels"] = batch["label"]
            batch["y"] = batch["label"]
            if "domain" in batch:
                batch["domain"] = batch["domain"].to(self.device)
        return batch

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
        start = 0
        from .datasets import _split_batch  # local import: avoid cycles at module load

        for raw in loader:
            images, labels = _split_batch(raw)
            yield self._to_batch(images, labels, start)
            start += int(labels.shape[0]) if torch.is_tensor(labels) else len(labels)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return int(math.ceil(n / float(self.batch_size)))

    # Convenience helper used by scripts that only need tensors.
    def image_batches(self) -> Iterator[torch.Tensor]:
        for batch in self:
            yield batch["image"]


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def build_label_shift_stream(
    cfg: Any = None,
    corruption: Optional[str] = None,
    corruption_root: Optional[str] = None,
    root: Optional[str] = None,
    dataset: str = "imagenet-c",
    severity: int = 5,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    image_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    mode: str = "class_order",
    decay: float = 0.98,
    dirichlet_alpha: float = 0.5,
    limit: Optional[int] = None,
    transform=None,
    **kwargs: Any,
) -> NonIIDStream:
    """Class-ordered ("online imbalanced label distribution") test stream.

    The dataset for ``corruption`` is built exactly as for the i.i.d. protocol and
    then re-ordered by ascending class label.  Set ``mode`` to ``"geometric"`` or
    ``"dirichlet"`` for an explicitly imbalanced sample budget per class.
    """
    root = root if root is not None else _cfg_get(cfg, "data", "root", default="./data")
    dataset_name = dataset or _cfg_get(cfg, "data", "dataset", default="imagenet-c")
    severity = int(severity if severity is not None else _cfg_get(cfg, "data", "severity", default=5))
    batch_size = int(batch_size if batch_size is not None else _cfg_get(cfg, "data", "batch_size", default=64))
    num_workers = int(num_workers if num_workers is not None else _cfg_get(cfg, "data", "num_workers", default=4))
    image_size = int(image_size if image_size is not None else _cfg_get(cfg, "data", "image_size", default=224))
    seed = int(seed if seed is not None else _cfg_get(cfg, "seed", default=0))
    corruption = corruption or _cfg_get(cfg, "data", "corruption", default=None)
    if limit is None:
        limit = _cfg_get(cfg, "data", "limit", default=None)

    base = build_dataset(
        dataset=dataset_name,
        root=corruption_root if corruption_root is not None else root,
        corruption=corruption,
        severity=severity,
        image_size=image_size,
        transform=transform,
        limit=limit,
        seed=seed,
        **kwargs,
    )
    labels = _dataset_labels(base)
    order = label_shift_indices(
        labels,
        seed=seed,
        mode=mode,
        decay=decay,
        dirichlet_alpha=dirichlet_alpha,
    )
    subset = Subset(base, order.tolist())
    domains = np.asarray([order_i for order_i in range(len(order))], dtype=np.int64)  # placeholder
    return NonIIDStream(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        dataset_name=dataset_name,
        scenario="label_shift",
        domains=domains,
        domain_names=[str(corruption)] if corruption else None,
        **{k: v for k, v in kwargs.items() if k in ("drop_last", "pin_memory", "collate_fn")},
    )


def build_mixed_domain_stream(
    cfg: Any = None,
    corruptions: Optional[Sequence[str]] = None,
    severity: int = 5,
    root: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    image_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    limit_per_domain: Optional[int] = None,
    transform=None,
    **kwargs: Any,
) -> NonIIDStream:
    """Mixed-domain stream: the 15 ImageNet-C corruptions concatenated in random order.

    A single pass over this stream emulates the "mixed domain shifts" scenario of
    Table 11 in which "test data stream consists of multiple randomly mixed
    domains with different distribution shifts".
    """
    root = root if root is not None else _cfg_get(cfg, "data", "root", default="./data")
    severity = int(severity if severity is not None else _cfg_get(cfg, "data", "severity", default=5))
    batch_size = int(batch_size if batch_size is not None else _cfg_get(cfg, "data", "batch_size", default=64))
    num_workers = int(num_workers if num_workers is not None else _cfg_get(cfg, "data", "num_workers", default=4))
    image_size = int(image_size if image_size is not None else _cfg_get(cfg, "data", "image_size", default=224))
    seed = int(seed if seed is not None else _cfg_get(cfg, "seed", default=0))
    if corruptions is None:
        corruptions = _cfg_get(cfg, "data", "corruptions", default=None)
    if corruptions is None:
        corruptions = list(IMAGENET_C_CORRUPTIONS)
    corruptions = list(corruptions)

    datasets = []
    for corruption in corruptions:
        ds = build_dataset(
            dataset="imagenet-c",
            root=root,
            corruption=corruption,
            severity=severity,
            image_size=image_size,
            transform=transform,
            limit=limit_per_domain,
            seed=seed,
            **kwargs,
        )
        datasets.append(ds)

    mixed = MixedDomainDataset(datasets, domain_names=corruptions, seed=seed, shuffle=True)
    return NonIIDStream(
        mixed,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        dataset_name="imagenet-c-mixed",
        scenario="mixed_shift",
        domains=mixed.domains,
        domain_names=list(mixed._names),
        **{k: v for k, v in kwargs.items() if k in ("drop_last", "pin_memory", "collate_fn")},
    )


def build_non_iid_stream(
    cfg: Any = None,
    scenario: str = "label_shift",
    corruption: Optional[str] = None,
    corruptions: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> NonIIDStream:
    """Config-driven dispatcher over the Table 11 non-i.i.d. scenarios.

    ``scenario`` must be one of :data:`NON_IID_SCENARIOS`; ``"iid"`` simply builds
    the ordinary ordered ImageNet-C stream of :mod:`src.data.datasets`.
    """
    scenario = str(scenario).lower().replace("-", "_")
    if scenario in ("label_shift", "label", "imbalanced", "online_imbalanced_label_distribution"):
        return build_label_shift_stream(cfg=cfg, corruption=corruption, **kwargs)
    if scenario in ("mixed_shift", "mixed", "mixed_domain", "mixed_domains"):
        return build_mixed_domain_stream(cfg=cfg, corruptions=corruptions, **kwargs)
    if scenario in ("iid", "mild", "none"):
        root = kwargs.pop("root", None) or _cfg_get(cfg, "data", "root", default="./data")
        batch_size = kwargs.pop("batch_size", None) or _cfg_get(cfg, "data", "batch_size", default=64)
        num_workers = kwargs.pop("num_workers", None) or _cfg_get(cfg, "data", "num_workers", default=4)
        image_size = kwargs.pop("image_size", None) or _cfg_get(cfg, "data", "image_size", default=224)
        severity = kwargs.pop("severity", None) or _cfg_get(cfg, "data", "severity", default=5)
        seed = kwargs.pop("seed", None) or _cfg_get(cfg, "seed", default=0)
        transform = kwargs.pop("transform", None)
        limit = kwargs.pop("limit", None)
        base = build_dataset(
            dataset="imagenet-c",
            root=root,
            corruption=corruption,
            severity=int(severity),
            image_size=int(image_size),
            transform=transform,
            limit=limit,
            seed=int(seed),
        )
        return NonIIDStream(
            base,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            device=kwargs.pop("device", None),
            dataset_name="imagenet-c",
            scenario="iid",
            domain_names=[str(corruption)] if corruption else None,
        )
    raise ValueError(
        "unknown non-i.i.d. scenario %r (expected one of %s)" % (scenario, NON_IID_SCENARIOS)
    )


def non_iid_summary_key(scenario: str, corruption: Optional[str] = None) -> str:
    """Reporting key used by ``run_non_iid.py`` (mirrors Table 11 layout)."""
    scenario = str(scenario).lower()
    if scenario in ("iid", "mild"):
        return "mild_iid"
    if scenario in ("label_shift", "label"):
        return "online_label_shift"
    if scenario in ("mixed_shift", "mixed"):
        return "mixed_shifts"
    return scenario
