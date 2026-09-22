"""Feature-space t-SNE analysis for SMM (paper Sec. 5, Figure 6).

The paper states:

    "Feature Space Visualization Results. Figure 6 shows the tSNE
     (Van der Maaten & Hinton, 2008) visualization results of the output
     layer feature before the label mapping layer. Before applying VR
     methods, the target domain's output feature space shows limited class
     separation. With the baseline methods, we observe enhanced but
     incomplete separations, where certain class pairs ... remain
     indistinguishable in the feature space. By applying f_mask, our method
     successfully resolves incorrectly clustered classes ..."

The addendum fixes the analysis protocol:

* extract the **output-layer features before the label-mapping layer**,
* use **5000 randomly selected training samples per dataset** (seeded),
* ResNet-18 is used as the pre-trained model in the example (Figure 6).

This module implements exactly that: for each method (``without_vr`` reference,
shared-mask VR baselines, and SMM/``ours``) it trains the reprogramming module
with the paper's schedule, extracts a fixed seeded 5000-sample feature bank from
the frozen classifier, computes a 2-D t-SNE embedding (scikit-learn), and
renders a Figure-6-style grid.

Out of scope per the reproduction plan/addendum: Figures 1, 2 and 6's mask /
shared-pattern visualization subsection are not reproduced here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants (paper Sec. 5 + addendum)
# ---------------------------------------------------------------------------

#: Addendum: "5000 randomly selected training samples per dataset".
TSNE_SAMPLES: int = 5000

#: Paper uses the standard t-SNE default perplexity.
DEFAULT_PERPLEXITY: float = 30.0

#: Seed for deterministic sample selection / t-SNE initialisation.
DEFAULT_SEED: int = 0

#: Name of the proposed method inside the runners.
SMM_METHOD: str = "ours"

#: Datasets shown in Figure 6.
TSNE_DATASETS: Tuple[str, ...] = ("svhn", "eurosat")

#: Shared-mask VR baselines (Table 1 / Table 2).
BASELINE_METHODS: Tuple[str, ...] = ("pad", "narrow", "medium", "full")

#: Label used for the "before applying VR methods" reference.
NO_VR_METHOD: str = "without_vr"

#: Plotting order: no-VR, weakest baseline, ..., SMM last (best separation).
METHOD_ORDER: Tuple[str, ...] = (NO_VR_METHOD,) + BASELINE_METHODS + (SMM_METHOD,)

#: Human-readable method labels.
METHOD_DISPLAY: Dict[str, str] = {
    NO_VR_METHOD: "w/o VR",
    "pad": "Pad",
    "narrow": "Narrow",
    "medium": "Medium",
    "full": "Full",
    SMM_METHOD: "SMM (ours)",
}

#: Default backbone (Figure 6 example).
DEFAULT_BACKBONE: str = "resnet18"

#: Default output locations.
DEFAULT_OUTPUT_DIR: str = "outputs/figures"
DEFAULT_FEATURE_FILE: str = "tsne_features.json"
DEFAULT_FIGURE_NAME: str = "figure6_tsne.pdf"

#: Pretty dataset names for titles.
DISPLAY_NAMES: Dict[str, str] = {
    "svhn": "SVHN",
    "eurosat": "EuroSAT",
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "gtsrb": "GTSRB",
    "flowers102": "Flowers102",
    "dtd": "DTD",
    "ucf101": "UCF101",
    "food101": "FOOD101",
    "sun397": "SUN397",
    "oxfordpets": "OxfordPets",
    "stanfordcars": "StanfordCars",
}

#: Training schedule shared with the main tables (fair comparison).
TRAINING_DEFAULTS: Dict[str, Any] = {
    "epochs": 200,
    "milestones": (100, 145),
    "lr": 0.01,
    "gamma": 0.1,
    "alpha_delta": 0.01,
    "gamma_delta": 0.1,
    "optimizer": "sgd",
    "momentum": 0.9,
    "batch_size": 256,
    "label_mapping": "ilm",
    "patch_size": 8,
}

# Reference accuracies (Table 1 / Table 2 averages) kept for reporting context.
REFERENCE_AVERAGES: Dict[str, Dict[str, float]] = {
    "resnet18": {"ours": 52.53, "full": 46.85, "medium": 45.04, "narrow": 43.48, "pad": 43.91},
    "resnet50": {"ours": 56.35, "full": 52.10, "medium": 49.39, "narrow": 46.76, "pad": 49.15},
    "vit_b32": {"ours": 72.4, "full": 64.7, "medium": 65.2, "narrow": 63.7, "pad": 53.1},
}

# ---------------------------------------------------------------------------
# Guarded project imports (module stays importable during partial builds)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - availability depends on build state
    from ..engine.seeds import SEEDS, resolve_seeds, set_seed
except Exception:  # pragma: no cover
    SEEDS = (0, 1, 2)

    def resolve_seeds(seeds=None, n_seeds=None):  # type: ignore
        if seeds is None:
            seeds = SEEDS if n_seeds in (None, 0) else tuple(range(n_seeds))
        return list(seeds)

    def set_seed(seed, **kwargs):  # type: ignore
        import random

        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        return seed


try:  # pragma: no cover
    from ..engine.metrics import RunResult, aggregate_seeds, format_mean_std
except Exception:  # pragma: no cover
    RunResult = None  # type: ignore

    def aggregate_seeds(values, ddof=1):  # type: ignore
        vals = [float(v) for v in values]
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        if len(vals) < 2:
            return mean, 0.0
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - ddof)
        return mean, math.sqrt(max(var, 0.0))

    def format_mean_std(mean, std, decimals=2):  # type: ignore
        return f"{mean:.{decimals}f} +- {std:.{decimals}f}"


try:  # pragma: no cover
    from ..data.datasets import DEFAULT_BATCH_SIZES, MAIN_DATASETS, build_dataloaders
except Exception:  # pragma: no cover
    MAIN_DATASETS = (
        "cifar10",
        "cifar100",
        "svhn",
        "gtsrb",
        "flowers102",
        "dtd",
        "ucf101",
        "food101",
        "sun397",
        "eurosat",
        "oxfordpets",
    )
    DEFAULT_BATCH_SIZES = {name: (64 if name in ("dtd", "oxfordpets") else 256) for name in MAIN_DATASETS}

    def build_dataloaders(*args, **kwargs):  # type: ignore
        raise RuntimeError("smm_vr.data.datasets is unavailable")


try:  # pragma: no cover
    from ..models.pretrained import build_classifier, input_size_for
except Exception:  # pragma: no cover
    build_classifier = None  # type: ignore

    def input_size_for(backbone="resnet18", imgsize=None):  # type: ignore
        if imgsize:
            return int(imgsize)
        return 384 if "vit" in str(backbone).lower() else 224


try:  # pragma: no cover
    from ..engine.evaluate import extract_features, extract_features_for_tsne
except Exception:  # pragma: no cover
    extract_features = None  # type: ignore
    extract_features_for_tsne = None  # type: ignore


try:  # pragma: no cover
    from ..engine.train_smm import train_smm, SMMTrainConfig, train_one_seed
except Exception:  # pragma: no cover
    train_smm = None  # type: ignore
    train_one_seed = None  # type: ignore
    SMMTrainConfig = None  # type: ignore


try:  # pragma: no cover
    from ..methods.baselines import (
        BaselineTrainConfig,
        build_shared_mask_baseline,
        train_baseline,
    )
except Exception:  # pragma: no cover
    BaselineTrainConfig = None  # type: ignore
    build_shared_mask_baseline = None  # type: ignore
    train_baseline = None  # type: ignore


try:  # pragma: no cover
    from ..modules.reprogram import build_smm_reprogram
except Exception:  # pragma: no cover
    build_smm_reprogram = None  # type: ignore


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def canonical_method(name: Optional[str]) -> str:
    """Normalise a method spelling to the canonical experiment key."""
    if name is None:
        return SMM_METHOD
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ours": SMM_METHOD,
        "smm": SMM_METHOD,
        "proposed": SMM_METHOD,
        "without_vr": NO_VR_METHOD,
        "no_vr": NO_VR_METHOD,
        "none": NO_VR_METHOD,
        "base": NO_VR_METHOD,
        "original": NO_VR_METHOD,
        "shared": "full",
        "full_watermark": "full",
    }
    return aliases.get(key, key)


def canonical_dataset(name: str) -> str:
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    table = {
        "cifar10": "cifar10",
        "cifar100": "cifar100",
        "svhn": "svhn",
        "gtsrb": "gtsrb",
        "flowers102": "flowers102",
        "dtd": "dtd",
        "ucf101": "ucf101",
        "food101": "food101",
        "sun397": "sun397",
        "eurosat": "eurosat",
        "oxfordpets": "oxfordpets",
        "oxfordiiitpet": "oxfordpets",
        "stanfordcars": "stanfordcars",
        "cars": "stanfordcars",
    }
    return table.get(key, str(name).strip().lower())


def canonical_backbone(name: Optional[str]) -> str:
    if not name:
        return DEFAULT_BACKBONE
    key = str(name).strip().lower().replace("-", "_")
    table = {
        "resnet18": "resnet18",
        "resnet_18": "resnet18",
        "r18": "resnet18",
        "resnet50": "resnet50",
        "resnet_50": "resnet50",
        "r50": "resnet50",
        "vit_b32": "vit_b32",
        "vitb32": "vit_b32",
        "vit_b_32": "vit_b32",
        "vit": "vit_b32",
        "vit_large": "vit_large",
        "vit_l_16": "vit_large",
    }
    if key in table:
        return table[key]
    return "vit_b32" if "vit" in key else "resnet18"


def baseline_methods(names: Optional[Sequence[str]] = None) -> List[str]:
    """Return the list of shared-mask baseline method names to plot."""
    if names is None:
        return [canonical_method(n) for n in BASELINE_METHODS]
    return [canonical_method(n) for n in names]


def method_order(methods: Optional[Sequence[str]] = None) -> List[str]:
    """Deterministic display order, always ending with SMM (``ours``)."""
    if methods is None:
        return list(METHOD_ORDER)
    canonical = [canonical_method(m) for m in methods]
    ordered = [m for m in METHOD_ORDER if m in canonical]
    ordered += [m for m in canonical if m not in ordered]
    # ensure SMM is last for a readable grid
    if SMM_METHOD in ordered:
        ordered.remove(SMM_METHOD)
        ordered.append(SMM_METHOD)
    return ordered


def batch_size_for(dataset: str) -> int:
    """Batch size from Table 9 (256, except DTD/OxfordPets = 64)."""
    name = canonical_dataset(dataset)
    try:
        return int(DEFAULT_BATCH_SIZES[name])
    except Exception:
        return 64 if name in ("dtd", "oxfordpets") else 256


def resolve_device(device: Optional[Any] = None) -> Any:
    import torch

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device) if isinstance(device, str) else device


def default_output_dir() -> str:
    return os.environ.get("SMM_FIGURES_DIR", DEFAULT_OUTPUT_DIR)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _select_indices(num_samples: int, total: int, seed: int) -> Any:
    """Deterministic random subset selection (addendum: 5000 samples/dataset)."""
    import torch

    if total <= 0:
        return torch.empty(0, dtype=torch.long)
    k = min(int(num_samples), int(total))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.randperm(int(total), generator=generator)[:k]


def extract_features_with_indices(
    classifier: Any,
    dataset: Any,
    *,
    f_in: Any = None,
    num_samples: int = TSNE_SAMPLES,
    seed: int = DEFAULT_SEED,
    batch_size: Optional[int] = None,
    num_workers: int = 4,
    device: Optional[Any] = None,
    indices: Optional[Any] = None,
) -> Tuple[Any, Any, Any]:
    """Extract output-layer features for a seeded subset of a dataset.

    Implements the addendum protocol: ``num_samples`` (default 5000) randomly
    selected *training* samples per dataset, features taken from the frozen
    classifier's output layer **before** the label mapping.

    Returns ``(features, labels, indices)``.
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    torch_device = resolve_device(device)
    if indices is None:
        indices = _select_indices(num_samples, len(dataset), seed)
    subset = Subset(dataset, [int(i) for i in indices.tolist()])
    loader = DataLoader(
        subset,
        batch_size=batch_size or min(256, max(1, len(subset))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    feats: List[Any] = []
    labels: List[Any] = []
    if extract_features is not None:
        features, targets = extract_features(
            classifier, loader, f_in=f_in, device=torch_device, normalize=False
        )
        return features.detach().cpu().float(), targets.detach().cpu().long(), indices

    # Local fallback: run the frozen classifier and use its logits as features.
    classifier.eval()
    with torch.no_grad():
        for batch in loader:
            images, target = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) else (batch, None)
            images = images.to(torch_device)
            if f_in is not None:
                images = f_in(images)
            out = classifier(images)
            if isinstance(out, (tuple, list)):
                out = out[0]
            feats.append(out.detach().cpu().float())
            if target is not None:
                labels.append(target.detach().cpu().long())
            else:
                labels.append(torch.full((out.shape[0],), -1, dtype=torch.long))
    features = torch.cat(feats, dim=0) if feats else torch.zeros(0, 1)
    targets = torch.cat(labels, dim=0) if labels else torch.zeros(0, dtype=torch.long)
    return features, targets, indices


def extract_feature_bank(
    dataset: str,
    *,
    methods: Optional[Sequence[str]] = None,
    backbone: str = DEFAULT_BACKBONE,
    seeds: Optional[Sequence[int]] = None,
    num_samples: int = TSNE_SAMPLES,
    num_workers: int = 4,
    data_root: Optional[str] = None,
    download: bool = True,
    device: Optional[Any] = None,
    patch_size: int = 8,
    label_mapping: str = "ilm",
    train_fraction: Optional[float] = None,
    split_seed: int = 0,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    verbose: bool = False,
    log: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """Train every method and extract a seeded feature bank per method.

    Returns ``{method: {"features": Tensor, "labels": Tensor, "indices": Tensor,
    "accuracy": float, "seed": int}}``. ``without_vr`` needs no training: it uses
    the raw transformed target images through the frozen classifier.
    """
    dataset = canonical_dataset(dataset)
    backbone = canonical_backbone(backbone)
    seeds = resolve_seeds(seeds)
    seed = int(seeds[0]) if seeds else DEFAULT_SEED
    device_obj = resolve_device(device)

    methods = method_order(methods)

    if build_classifier is None or build_dataloaders is None:
        raise RuntimeError(
            "smm_vr.models.pretrained / smm_vr.data.datasets are required for t-SNE feature extraction"
        )

    imgsize = input_size_for(backbone)
    classifier = build_classifier(backbone=backbone, device=device_obj)
    num_classes = getattr(classifier, "num_classes", None) or getattr(classifier, "out_features", None)

    train_loader, test_loader, _spec = build_dataloaders(
        dataset,
        backbone=backbone,
        data_root=data_root,
        imgsize=imgsize,
        batch_size=batch_size_for(dataset),
        num_workers=num_workers,
        download=download,
        device=str(device_obj),
    )
    train_dataset = getattr(train_loader, "dataset", None)
    if train_dataset is None:
        raise RuntimeError("training DataLoader does not expose its dataset")

    # Fixed subset shared by all methods so the comparison is apples-to-apples.
    indices = _select_indices(num_samples, len(train_dataset), seed)

    bank: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        if verbose:
            print(f"[tsne] extracting features: dataset={dataset} method={method}")
        if method == NO_VR_METHOD:
            features, labels, _ = extract_features_with_indices(
                classifier,
                train_dataset,
                f_in=None,
                num_samples=num_samples,
                seed=seed,
                num_workers=num_workers,
                device=device_obj,
                indices=indices,
            )
            accuracy = None
            if extract_features_for_tsne is not None and test_loader is not None:
                # Optional: quick no-VR accuracy for context (not required by Fig. 6).
                accuracy = None
            bank[method] = {
                "features": features,
                "labels": labels,
                "indices": indices,
                "accuracy": accuracy,
                "seed": seed,
            }
            continue

        model, accuracy = train_method_for_features(
            dataset=dataset,
            method=method,
            backbone=backbone,
            classifier=classifier,
            train_loader=train_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            seed=seed,
            device=device_obj,
            patch_size=patch_size,
            label_mapping=label_mapping,
            num_workers=num_workers,
            data_root=data_root,
            download=download,
            train_fraction=train_fraction,
            split_seed=split_seed,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
            verbose=verbose,
            log=log,
        )

        features, labels, _ = extract_features_with_indices(
            classifier,
            train_dataset,
            f_in=model,
            num_samples=num_samples,
            seed=seed,
            num_workers=num_workers,
            device=device_obj,
            indices=indices,
        )
        bank[method] = {
            "features": features,
            "labels": labels,
            "indices": indices,
            "accuracy": accuracy,
            "seed": seed,
        }
    return bank


def train_method_for_features(
    *,
    dataset: str,
    method: str,
    backbone: str,
    classifier: Any,
    train_loader: Any,
    test_loader: Any,
    num_classes: Optional[int],
    seed: int,
    device: Any,
    patch_size: int = 8,
    label_mapping: str = "ilm",
    num_workers: int = 4,
    data_root: Optional[str] = None,
    download: bool = True,
    train_fraction: Optional[float] = None,
    split_seed: int = 0,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    verbose: bool = False,
    log: Any = None,
) -> Tuple[Any, Optional[float]]:
    """Train one method with the paper schedule, returning ``(f_in, accuracy)``."""
    method = canonical_method(method)
    set_seed(seed)

    common = dict(
        epochs=int(TRAINING_DEFAULTS["epochs"]),
        milestones=tuple(TRAINING_DEFAULTS["milestones"]),
        alpha_delta=float(TRAINING_DEFAULTS["alpha_delta"]),
        gamma_delta=float(TRAINING_DEFAULTS["gamma_delta"]),
        optimizer=str(TRAINING_DEFAULTS["optimizer"]),
        momentum=float(TRAINING_DEFAULTS["momentum"]),
        batch_size=batch_size_for(dataset),
        num_workers=num_workers,
        label_mapping=label_mapping,
        seed=seed,
        device=device,
        verbose=verbose,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
    )

    if method == SMM_METHOD:
        if train_smm is None or build_smm_reprogram is None:
            raise RuntimeError("SMM training requires engine.train_smm and modules.reprogram")
        model = build_smm_reprogram(backbone=backbone, patch_size=patch_size)
        model = model.to(device)
        config = _make_smm_config(dataset, backbone, common)
        history = train_smm(model, classifier, train_loader, test_loader=test_loader, config=config, dataset=dataset, device=device, logger=log)
        accuracy = getattr(history, "final_test_accuracy", None)
        if accuracy is None:
            accuracy = getattr(history, "best_test_accuracy", None)
        return model, accuracy

    # shared-mask VR baseline (Pad / Narrow / Medium / Full)
    if train_baseline is None or build_shared_mask_baseline is None:
        raise RuntimeError("baseline training requires methods.baselines")
    model = build_shared_mask_baseline(method, backbone=backbone)
    model = model.to(device)
    config = _make_baseline_config(dataset, method, backbone, common)
    history = train_baseline(model, classifier, train_loader, test_loader=test_loader, config=config, dataset=dataset, device=device, logger=log)
    accuracy = getattr(history, "final_test_accuracy", None)
    if accuracy is None:
        accuracy = getattr(history, "best_test_accuracy", None)
    return model, accuracy


def _filter_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import dataclasses

        allowed = {f.name for f in dataclasses.fields(cls)}
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return dict(kwargs)


def _make_smm_config(dataset: str, backbone: str, common: Dict[str, Any]) -> Any:
    if SMMTrainConfig is None:
        return None
    kwargs = dict(common)
    kwargs["backbone"] = backbone
    kwargs["patch_size"] = int(TRAINING_DEFAULTS["patch_size"])
    return SMMTrainConfig(**_filter_kwargs(SMMTrainConfig, kwargs))


def _make_baseline_config(dataset: str, name: str, backbone: str, common: Dict[str, Any]) -> Any:
    if BaselineTrainConfig is None:
        return None
    kwargs = dict(common)
    kwargs["name"] = name
    kwargs["backbone"] = backbone
    kwargs.pop("alpha_delta", None)
    kwargs.pop("gamma_delta", None)
    kwargs["lr"] = float(TRAINING_DEFAULTS["lr"])
    kwargs["gamma"] = float(TRAINING_DEFAULTS["gamma"])
    return BaselineTrainConfig(**_filter_kwargs(BaselineTrainConfig, kwargs))


# ---------------------------------------------------------------------------
# t-SNE computation
# ---------------------------------------------------------------------------


def compute_tsne(
    features: Any,
    *,
    perplexity: float = DEFAULT_PERPLEXITY,
    seed: int = DEFAULT_SEED,
    max_iter: Optional[int] = None,
    n_components: int = 2,
    verbose: bool = False,
) -> Any:
    """Compute a 2-D t-SNE embedding (Van der Maaten & Hinton, 2008).

    Falls back to a deterministic PCA projection when scikit-learn is missing so
    that the analysis still produces a usable (if approximate) figure.
    """
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for t-SNE analysis") from exc

    X = features
    try:
        import torch

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
    except Exception:
        pass
    X = np.asarray(X, dtype=np.float64)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if X.shape[0] == 0:
        return np.zeros((0, n_components), dtype=np.float64)

    # perplexity must be < n_samples
    n = X.shape[0]
    perp = float(min(float(perplexity), max(5.0, (n - 1) / 3.0))) if n > 6 else max(2.0, (n - 1) / 2.0)
    perp = max(2.0, min(perp, n - 1)) if n > 3 else 1.0

    try:
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=n_components,
            perplexity=perp,
            random_state=int(seed),
            init="pca" if n > 10 else "random",
            learning_rate="auto",
            max_iter=max_iter or 1000,
            verbose=1 if verbose else 0,
        )
        return tsne.fit_transform(X)
    except Exception as exc:  # pragma: no cover - sklearn optional
        if verbose:
            print(f"[tsne] scikit-learn unavailable ({exc}); using PCA fallback")
        Xc = X - X.mean(axis=0, keepdims=True)
        try:
            _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
            comps = vt[:n_components].T
        except Exception:
            comps = np.zeros((Xc.shape[1], n_components))
            comps[: min(n_components, Xc.shape[1])] = np.eye(min(n_components, Xc.shape[1]))
        return Xc @ comps


#: Convenience alias used by ``analysis/__init__.py``.
tsne_embedding = compute_tsne


def embedding_for_bank(
    bank: Dict[str, Dict[str, Any]],
    *,
    perplexity: float = DEFAULT_PERPLEXITY,
    seed: int = DEFAULT_SEED,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Compute embeddings for every method entry of a feature bank."""
    out: Dict[str, Dict[str, Any]] = {}
    for method, entry in bank.items():
        emb = compute_tsne(entry["features"], perplexity=perplexity, seed=seed, verbose=verbose)
        out[method] = {
            "embedding": emb,
            "labels": _to_list(entry.get("labels")),
            "accuracy": entry.get("accuracy"),
        }
    return out


def _to_list(value: Any, limit: Optional[int] = None) -> List[Any]:
    if value is None:
        return []
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return [v for v in list(value)[: limit or len(value)]]
    try:
        return [v for v in list(value)[: limit or len(value)]]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Reporting / plotting
# ---------------------------------------------------------------------------


def summarise_embeddings(
    embeddings: Dict[str, Dict[str, Any]],
    *,
    method: str = SMM_METHOD,
    method_a: Optional[str] = None,
    method_b: Optional[str] = None,
) -> Dict[str, Any]:
    """Quantify class separation (mean within/between-class distances).

    The paper describes the effect qualitatively ("resolves incorrectly
    clustered classes"); this provides a numerical sanity check for the same
    claim using the paper's own metric-free language.
    """
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return {}

    def _score(emb: Any, labels: Any) -> Optional[float]:
        if emb is None or not len(emb) or not len(labels):
            return None
        E = np.asarray(emb, dtype=np.float64)
        L = np.asarray(labels)
        classes = sorted(set(L.tolist()))
        if len(classes) < 2:
            return None
        centroids = np.stack([E[L == c].mean(axis=0) for c in classes])
        within, counts = 0.0, 0
        for idx, c in enumerate(classes):
            pts = E[L == c]
            within += float(np.linalg.norm(pts - centroids[idx], axis=1).sum())
            counts += pts.shape[0]
        within /= max(counts, 1)
        between = []
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                between.append(float(np.linalg.norm(centroids[i] - centroids[j])))
        if not between:
            return None
        return float(np.mean(between)) / max(within, 1e-8)

    summary: Dict[str, Any] = {}
    for name, entry in embeddings.items():
        summary[name] = {
            "separation_ratio": _score(entry.get("embedding"), entry.get("labels")),
            "accuracy": entry.get("accuracy"),
        }
    if method_a and method_b and method_a in summary and method_b in summary:
        a = summary[method_a]["separation_ratio"]
        b = summary[method_b]["separation_ratio"]
        if a is not None and b is not None:
            summary["improved"] = bool(b > a)
    return summary


def plot_tsne_grid(
    embeddings: Dict[str, Dict[str, Any]],
    *,
    output_path: Optional[str] = DEFAULT_FIGURE_NAME,
    datasets: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    dpi: int = 200,
    figsize: Optional[Tuple[float, float]] = None,
    show_accuracy: bool = True,
    save: bool = True,
) -> Optional[str]:
    """Render a Figure-6-style grid of t-SNE scatter plots.

    ``embeddings`` may be either a single-dataset mapping ``{method: entry}`` or
    a nested mapping ``{dataset: {method: entry}}``. matplotlib is optional: if it
    is unavailable a JSON sidecar with the coordinates is written instead.
    """
    matplotlib = _get_matplotlib()
    if matplotlib is None:
        if output_path and save:
            fallback = _with_suffix(output_path, ".json")
            with open(fallback, "w") as handle:
                json.dump(_jsonable(embeddings), handle)
            return fallback
        return None

    import numpy as np
    import matplotlib.pyplot as plt

    # Normalise to {dataset: {method: entry}}
    if datasets is None:
        if all(isinstance(v, dict) and "embedding" in v for v in embeddings.values()):
            grouped: Dict[str, Dict[str, Any]] = {"": embeddings}  # type: ignore
        else:
            grouped = embeddings  # type: ignore
    else:
        grouped = {d: embeddings.get(d, {}) for d in datasets}  # type: ignore
    grouped = {k: v for k, v in grouped.items() if v}

    if not grouped:
        return None

    panel_datasets = list(grouped.keys())
    methods = method_order([m for panel in grouped.values() for m in panel.keys()])
    n_rows = len(panel_datasets)
    n_cols = max(1, len(methods))
    if figsize is None:
        figsize = (3.1 * n_cols, 3.0 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    cmap = plt.get_cmap("tab20")

    for row, ds in enumerate(panel_datasets):
        for col, method in enumerate(methods):
            ax = axes[row][col]
            entry = grouped[ds].get(method)
            if not entry:
                ax.axis("off")
                continue
            emb = np.asarray(entry.get("embedding"))
            labels = np.asarray(entry.get("labels"))
            if emb.size == 0:
                ax.axis("off")
                continue
            if labels.size == emb.shape[0]:
                for c in sorted(set(labels.tolist())):
                    mask = labels == c
                    ax.scatter(
                        emb[mask, 0],
                        emb[mask, 1],
                        s=2.5,
                        color=cmap(int(c) % 20),
                        alpha=0.8,
                        linewidths=0,
                    )
            else:
                ax.scatter(emb[:, 0], emb[:, 1], s=2.5, alpha=0.8, linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])
            acc = entry.get("accuracy")
            label = METHOD_DISPLAY.get(method, method)
            if ds and row == 0:
                ax.set_title(label, fontsize=9)
            elif not ds:
                ax.set_title(label, fontsize=9)
            if show_accuracy and acc is not None:
                ax.set_xlabel(f"acc {float(acc):.1f}", fontsize=7)
        if ds:
            axes[row][0].set_ylabel(DISPLAY_NAMES.get(ds, ds), fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    if not (output_path and save):
        plt.close(fig)
        return None
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_figure6(
    embeddings: Dict[str, Dict[str, Any]],
    *,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    datasets: Sequence[str] = TSNE_DATASETS,
    save: bool = True,
    **kwargs: Any,
) -> Optional[str]:
    """Alias rendering Figure 6 (SVHN + EuroSAT by default)."""
    if output_path is None:
        output_dir = output_dir or default_output_dir()
        output_path = os.path.join(output_dir, DEFAULT_FIGURE_NAME)
    return plot_tsne_grid(embeddings, output_path=output_path, datasets=datasets, save=save, **kwargs)


def _get_matplotlib() -> Any:
    try:  # pragma: no cover - optional dependency
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # noqa: F401

        return matplotlib
    except Exception:
        return None


def _with_suffix(path: str, suffix: str) -> str:
    root, _ext = os.path.splitext(path)
    return root + suffix


def _jsonable(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
    except Exception:
        pass
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def save_feature_bank(path: str, bank: Dict[str, Dict[str, Any]]) -> str:
    """Persist a feature bank (embeddings + labels + accuracies) as JSON."""
    payload = {
        method: {
            "features": _jsonable(entry.get("features")),
            "labels": _jsonable(entry.get("labels")),
            "indices": _jsonable(entry.get("indices")),
            "accuracy": entry.get("accuracy"),
            "seed": entry.get("seed"),
        }
        for method, entry in bank.items()
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle)
    return path


# ---------------------------------------------------------------------------
# Experiment entry point
# ---------------------------------------------------------------------------


def describe_tsne_study(
    datasets: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    backbone: str = DEFAULT_BACKBONE,
    num_samples: int = TSNE_SAMPLES,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    datasets = [canonical_dataset(d) for d in (datasets or TSNE_DATASETS)]
    methods = method_order(methods)
    return {
        "experiment": "feature_space_tsne",
        "figure": "Figure 6",
        "backbone": canonical_backbone(backbone),
        "datasets": datasets,
        "methods": methods,
        "samples_per_dataset": int(num_samples),
        "perplexity": float(DEFAULT_PERPLEXITY),
        "seeds": resolve_seeds(seeds),
        "note": (
            "Features are the output-layer activations before the label mapping, "
            "extracted from a seeded random subset of the training split "
            "(addendum). Figures 1, 2 and the mask visualization subsection are "
            "out of scope for this reproduction."
        ),
    }


def run_tsne_experiment(
    datasets: Optional[Sequence[str]] = None,
    *,
    methods: Optional[Sequence[str]] = None,
    backbone: str = DEFAULT_BACKBONE,
    seeds: Optional[Sequence[int]] = None,
    num_samples: int = TSNE_SAMPLES,
    perplexity: float = DEFAULT_PERPLEXITY,
    output_dir: Optional[str] = None,
    feature_file: str = DEFAULT_FEATURE_FILE,
    figure_file: str = DEFAULT_FIGURE_NAME,
    save: bool = True,
    verbose: bool = True,
    num_workers: int = 4,
    data_root: Optional[str] = None,
    download: bool = True,
    device: Optional[Any] = None,
    patch_size: int = 8,
    label_mapping: str = "ilm",
    train_fraction: Optional[float] = None,
    split_seed: int = 0,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    log: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Reproduce the feature-space t-SNE study (paper Sec. 5, Figure 6).

    For every dataset it trains the selected methods, extracts a seeded
    5000-sample feature bank from the frozen classifier (before the label
    mapping), computes t-SNE embeddings and renders the comparison grid.
    """
    datasets = [canonical_dataset(d) for d in (datasets or TSNE_DATASETS)]
    methods = method_order(methods)
    backbone = canonical_backbone(backbone)
    output_dir = output_dir or default_output_dir()
    if save:
        os.makedirs(output_dir, exist_ok=True)

    description = describe_tsne_study(datasets, methods, backbone, num_samples, seeds)
    payload: Dict[str, Any] = {
        "experiment": "feature_space_tsne",
        "description": description,
        "datasets": {},
    }

    for dataset in datasets:
        if verbose:
            print(f"[tsne] dataset={dataset} backbone={backbone} methods={methods}")
        bank = extract_feature_bank(
            dataset,
            methods=methods,
            backbone=backbone,
            seeds=seeds,
            num_samples=num_samples,
            num_workers=num_workers,
            data_root=data_root,
            download=download,
            device=device,
            patch_size=patch_size,
            label_mapping=label_mapping,
            train_fraction=train_fraction,
            split_seed=split_seed,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
            verbose=verbose,
            log=log,
        )
        embeddings = embedding_for_bank(bank, perplexity=perplexity, seed=description["seeds"][0], verbose=verbose)
        summary = summarise_embeddings(embeddings, method_a=NO_VR_METHOD, method_b=SMM_METHOD)

        if save:
            feature_path = os.path.join(output_dir, f"{dataset}_{feature_file}")
            save_feature_bank(feature_path, bank)
        else:
            feature_path = None

        payload["datasets"][dataset] = {
            "methods": list(bank.keys()),
            "summary": summary,
            "feature_path": feature_path,
            "embeddings": _jsonable(embeddings),
        }
        if verbose:
            sep = {m: summary.get(m, {}).get("separation_ratio") for m in summary if isinstance(summary.get(m), dict)}
            print(f"[tsne] {dataset} separation ratios: {sep}")

    if save:
        figure_path = os.path.join(output_dir, figure_file)
        rendered = plot_figure6(
            {ds: {m: {"embedding": e["embedding"], "labels": e["labels"], "accuracy": e["accuracy"]} for m, e in data["embeddings"].items()} for ds, data in payload["datasets"].items()},
            output_path=figure_path,
            datasets=datasets,
            title=f"t-SNE feature space ({backbone})",
            save=True,
        )
        payload["figure_path"] = rendered

    return payload


#: Registry alias expected by ``smm_vr.experiments``/``analysis/__init__.py``.
def run_tsne_features(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return run_tsne_experiment(*args, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_tsne_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SMM feature-space t-SNE analysis (paper Sec. 5, Figure 6)",
    )
    parser.add_argument("--datasets", nargs="+", default=list(TSNE_DATASETS))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--samples", type=int, default=TSNE_SAMPLES, help="samples per dataset (addendum: 5000)")
    parser.add_argument("--perplexity", type=float, default=DEFAULT_PERPLEXITY)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--feature-file", default=DEFAULT_FEATURE_FILE)
    parser.add_argument("--figure-file", default=DEFAULT_FIGURE_NAME)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--label-mapping", default="ilm")
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--describe", action="store_true", help="print the study description and exit")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


#: Alias for symmetry with the other analysis modules.
build_arg_parser = build_tsne_arg_parser


def tsne_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_tsne_arg_parser()
    args = parser.parse_args(argv)

    if args.describe:
        print(json.dumps(describe_tsne_study(args.datasets, args.methods, args.backbone, args.samples, args.seeds), indent=2))
        return 0

    payload = run_tsne_experiment(
        args.datasets,
        methods=args.methods,
        backbone=args.backbone,
        seeds=args.seeds,
        num_samples=args.samples,
        perplexity=args.perplexity,
        output_dir=args.output_dir,
        feature_file=args.feature_file,
        figure_file=args.figure_file,
        save=not args.no_save,
        verbose=not args.quiet,
        num_workers=args.num_workers,
        data_root=args.data_root,
        device=args.device,
        patch_size=args.patch_size,
        label_mapping=args.label_mapping,
        train_fraction=args.train_fraction,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    if not args.quiet:
        print(json.dumps({ds: data["summary"] for ds, data in payload["datasets"].items()}, indent=2))
        if payload.get("figure_path"):
            print(f"saved figure: {payload['figure_path']}")
    return 0


#: Alias used by ``smm_vr.main`` dispatch.
main = tsne_main
tsne_main_entry = tsne_main


__all__ = [
    # constants
    "TSNE_SAMPLES",
    "DEFAULT_PERPLEXITY",
    "DEFAULT_SEED",
    "SMM_METHOD",
    "TSNE_DATASETS",
    "BASELINE_METHODS",
    "NO_VR_METHOD",
    "METHOD_ORDER",
    "METHOD_DISPLAY",
    "DEFAULT_BACKBONE",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_FEATURE_FILE",
    "DEFAULT_FIGURE_NAME",
    "DISPLAY_NAMES",
    "TRAINING_DEFAULTS",
    "REFERENCE_AVERAGES",
    # helpers
    "canonical_method",
    "canonical_dataset",
    "canonical_backbone",
    "baseline_methods",
    "method_order",
    "batch_size_for",
    "resolve_device",
    "default_output_dir",
    # feature extraction
    "extract_features_with_indices",
    "extract_feature_bank",
    "train_method_for_features",
    # t-SNE
    "compute_tsne",
    "tsne_embedding",
    "embedding_for_bank",
    "summarise_embeddings",
    # plotting / reporting
    "plot_tsne_grid",
    "plot_figure6",
    "save_feature_bank",
    "describe_tsne_study",
    # experiment
    "run_tsne_experiment",
    "run_tsne_features",
    # CLI
    "build_tsne_arg_parser",
    "build_arg_parser",
    "tsne_main",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(tsne_main())
