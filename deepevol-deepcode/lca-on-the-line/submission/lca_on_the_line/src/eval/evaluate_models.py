"""Model evaluation driver for the LCA-on-the-Line study (paper Section 4.1).

This module produces the 75-model (36 VMs + 39 VLMs) score table that is the
basis of Table 1, Table 2 (correlations) and Table 8 (LCA / ELCA / Top1).

For every model it computes, on the ImageNet in-distribution (ID) validation
set and on each OOD dataset (ImageNet-v2 MatchedFrequency, ImageNet-Sketch,
ImageNet-R, ImageNet-A, ObjectNet):

* Top-1 accuracy (``Top1``)
* Top-5 accuracy (``Top5``)
* ID LCA (``LCA``): mean information-content LCA distance over the
  misclassified samples, ``D_LCA(model, M) = (1/n) sum_i D_LCA(y_hat_i, y_i)``
  (Section D.2.1), and ``lca_all`` (the same sum normalised by ``n``).
* ELCA (``ELCA``): ``D_ELCA(model, M) = (1/(nK)) sum_i sum_k p_hat_{k,i}
  D_LCA(k, y_i)`` (Section D.3, Eq. 1).

The heavy part of the computation - a forward pass over the full ID/OOD
datasets - is cached to disk (``.npz`` with ``features``, ``logits`` and
``targets``) so that the later stages (latent hierarchies, soft-label probes,
prompt engineering) can reuse the penultimate features ``M(X)`` without
recomputing them.

The paper's Table 8 reference values (for the four model/dataset points shown)
are reproduced in :data:`TABLE8_REFERENCE` and can be used to sanity check a
full run::

    ResNet18      ID 6.643 / 7.505 / 0.698,  ImgN-v2 6.918 / 7.912 / 0.573 ...
    CLIP_RN50x4   ID 6.166 / 9.473 / 0.641,  ImgN-v2 6.383 / 9.525 / 0.573 ...

Note (paper Section D.3): ELCA must not be compared across modalities because
it is sensitive to the logit temperature, so comparisons are kept within a
family.

Source: Section 4.1, Table 1, Table 2, Table 8, Section D.3.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..data.imagenet import (  # noqa: F401 (re-exported for convenience)
    build_imagenet_dataset,
    build_imagenet_loader,
)
from ..data.ood_datasets import (
    DEFAULT_OOD_DATASETS,
    build_all_ood_datasets,
    build_ood_dataset,
    display_name,
    normalize_ood_name,
)
from ..metrics.lca_metric import (
    ModelMetrics,
    evaluate_model_outputs,
    softmax,
    top1_accuracy,
    top5_accuracy,
)
from ..hierarchy.lca_matrix import LcaMatrixProcessor, build_lca_matrix, process_lca_matrix
from ..hierarchy.wordnet import WordNetHierarchy, build_wordnet_hierarchy

logger = logging.getLogger(__name__)

__all__ = [
    "ModelRecord",
    "EvaluationConfig",
    "TABLE8_REFERENCE",
    "evaluate_model_on_dataset",
    "evaluate_model",
    "evaluate_zoo",
    "evaluate_zoo_from_cache",
    "collect_outputs",
    "records_to_rows",
    "records_to_dataframe",
    "save_results",
    "load_results",
    "summary_table",
    "build_hierarchy",
    "build_evaluation_loaders",
    "main",
]


# ---------------------------------------------------------------------------
# Paper reference values (Table 8) used for sanity checks / gating
# ---------------------------------------------------------------------------

#: ``model -> dataset -> (LCA, ELCA, Top1)`` from Table 8 of the paper.
TABLE8_REFERENCE: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "resnet18": {
        "ID": (6.643, 7.505, 0.698),
        "v2": (6.918, 7.912, 0.573),
        "s": (8.005, 9.283, 0.202),
        "r": (8.775, 8.853, 0.330),
        "a": (8.449, 9.622, 0.011),
        "objectnet": (8.062, 8.636, 0.272),
    },
    "resnet50": {
        "ID": (6.539, 7.012, 0.733),
        "v2": (6.863, 7.532, 0.610),
        "s": (7.902, 9.147, 0.235),
        "r": (8.779, 8.668, 0.361),
        "a": (8.424, 9.589, 0.018),
        "objectnet": (8.029, 8.402, 0.316),
    },
    "clip_rn50": {
        "ID": (6.327, 9.375, 0.579),
        "v2": (6.538, 9.442, 0.511),
        "s": (6.775, 9.541, 0.332),
        "r": (7.764, 9.127, 0.562),
        "a": (7.861, 9.526, 0.218),
        "objectnet": (7.822, 8.655, 0.398),
    },
    "clip_rn50x4": {
        "ID": (6.166, 9.473, 0.641),
        "v2": (6.383, 9.525, 0.573),
        "s": (6.407, 9.518, 0.415),
        "r": (7.435, 8.982, 0.681),
        "a": (7.496, 9.388, 0.384),
        "objectnet": (7.729, 8.354, 0.504),
    },
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvaluationConfig:
    """Knobs for :func:`evaluate_zoo`."""

    dataset_root: Optional[str] = None
    ood_roots: Dict[str, str] = field(default_factory=dict)
    cache_dir: Optional[str] = None
    batch_size: int = 256
    num_workers: int = 4
    resolution: int = 224
    temperature: float = 1.0
    max_samples: Optional[int] = None
    max_batches: Optional[int] = None
    device: Optional[str] = None
    allow_synthetic: bool = False
    overwrite_cache: bool = False
    compute_elca: bool = True
    ood_names: List[str] = field(default_factory=lambda: list(DEFAULT_OOD_DATASETS))
    sanity_tolerance: float = 0.15

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "EvaluationConfig":
        payload = dict(payload or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in payload.items() if k in known}
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ModelRecord:
    """One row of the 75-model score table (Table 1 / Table 8)."""

    name: str
    family: str = "vm"
    feature_dim: Optional[int] = None
    id_top1: float = float("nan")
    id_top5: float = float("nan")
    id_lca: float = float("nan")
    id_elca: float = float("nan")
    ood: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_id: int = 0
    runtime: float = 0.0
    error: Optional[str] = None

    # -- helpers -----------------------------------------------------------
    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

    def get(self, dataset: str, key: str, default: float = float("nan")) -> float:
        """Fetch an OOD metric, accepting dataset aliases (``v2``, ``S``, ...)."""
        key = {"top1": "top1", "top5": "top5", "lca": "lca", "elca": "elca"}.get(
            key.lower(), key
        )
        if normalize_ood_name(dataset) in ("id", "imagenet"):
            return {
                "top1": self.id_top1,
                "top5": self.id_top5,
                "lca": self.id_lca,
                "elca": self.id_elca,
            }.get(key, default)
        entry = self.ood.get(normalize_ood_name(dataset))
        if entry is None:
            for k, v in self.ood.items():
                if normalize_ood_name(k) == normalize_ood_name(dataset):
                    entry = v
                    break
        if entry is None:
            return default
        return float(entry.get(key, default))

    def row(self, datasets: Optional[Sequence[str]] = None) -> Dict[str, float]:
        """Flattened dict for tabular output: ``{dataset_top1: ...}``."""
        datasets = list(datasets) if datasets is not None else list(self.ood)
        out: Dict[str, float] = {
            "ID_Top1": self.id_top1,
            "ID_Top5": self.id_top5,
            "ID_LCA": self.id_lca,
            "ID_ELCA": self.id_elca,
        }
        for ds in datasets:
            tag = display_name(ds)
            out[f"{tag}_Top1"] = self.get(ds, "top1")
            out[f"{tag}_Top5"] = self.get(ds, "top5")
            out[f"{tag}_LCA"] = self.get(ds, "lca")
            out[f"{tag}_ELCA"] = self.get(ds, "elca")
        return out

    def check_against_table8(self, tolerance: float = 0.15) -> Dict[str, Dict[str, float]]:
        """Compare against the paper's Table 8 values (returns per-field deltas)."""
        ref = TABLE8_REFERENCE.get(self.name.lower())
        if ref is None:
            return {}
        diffs: Dict[str, Dict[str, float]] = {}
        for ds, (lca, elca, top1) in ref.items():
            measured = (
                (self.id_lca, self.id_elca, self.id_top1)
                if ds == "ID"
                else (self.get(ds, "lca"), self.get(ds, "elca"), self.get(ds, "top1"))
            )
            diffs[ds] = {
                "lca_delta": float(measured[0] - lca),
                "elca_delta": float(measured[1] - elca),
                "top1_delta": float(measured[2] - top1),
                "within_tolerance": float(
                    abs(measured[0] - lca) <= tolerance
                    and abs(measured[2] - top1) <= tolerance
                ),
            }
        return diffs


# ---------------------------------------------------------------------------
# Collecting model outputs (features, logits, targets)
# ---------------------------------------------------------------------------


def collect_outputs(
    model: Any,
    loader: Iterable,
    device: Optional[Any] = None,
    max_batches: Optional[int] = None,
    want_features: bool = True,
    desc: str = "",
) -> Dict[str, np.ndarray]:
    """Forward pass over ``loader`` collecting ``logits`` (and ``features``).

    Works with :class:`~src.models.vm_zoo.VisionModelWrapper` (``forward_both``)
    and with VLM wrappers exposing ``forward_both``/``logits``/``features``;
    falls back to calling ``model(x)`` when it returns a ``(features, logits)``
    tuple.
    """
    import torch  # local import: keep module importable without torch

    was_training = None
    if hasattr(model, "eval"):
        model.eval()
    if device is None:
        if hasattr(model, "device") and getattr(model, "device") is not None:
            device = model.device
        elif hasattr(model, "model"):
            try:
                device = next(model.model.parameters()).device  # type: ignore[arg-type]
            except (StopIteration, AttributeError, TypeError):
                device = "cpu"
        else:
            device = "cpu"

    iterator: Iterable = loader
    if max_batches is not None:
        iterator = _take(max_batches, loader)
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc=desc or "extract", leave=False)
    except Exception:  # pragma: no cover - tqdm optional
        pass

    feats: List[np.ndarray] = []
    logits: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for batch in iterator:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:  # dict-like / mapping batch
                x, y = batch["image"], batch["label"]
            x = x.to(device) if hasattr(x, "to") else x
            out_f = out_l = None
            if hasattr(model, "forward_both"):
                out_f, out_l = model.forward_both(x)
            elif hasattr(model, "logits") and hasattr(model, "features"):
                out_l = model.logits(x)
                try:
                    out_f = model.features(x)
                except Exception:
                    out_f = None
            else:
                out = model(x)
                if isinstance(out, (tuple, list)) and len(out) == 2:
                    out_f, out_l = out
                else:
                    out_l = out
            if out_l is not None:
                logits.append(_to_numpy(out_l).astype(np.float64))
            if want_features and out_f is not None:
                feats.append(_to_numpy(out_f).reshape(len(x), -1).astype(np.float64))
            targets.append(_to_numpy(y).astype(np.int64).reshape(-1))

    result: Dict[str, np.ndarray] = {
        "targets": np.concatenate(targets, axis=0) if targets else np.zeros((0,), np.int64),
    }
    result["logits"] = (
        np.concatenate(logits, axis=0) if logits else np.zeros((0, 0), np.float64)
    )
    if want_features:
        result["features"] = (
            np.concatenate(feats, axis=0) if feats else np.zeros((0, 0), np.float64)
        )
    if was_training is not None and hasattr(model, "train"):
        model.train(was_training)
    return result


def _take(n: int, iterable: Iterable) -> Iterable:
    for i, item in enumerate(iterable):
        if i >= n:
            return
        yield item


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):  # torch tensor
        return value.detach().cpu().numpy()
    return np.asarray(value)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _cache_file(cache_dir: Optional[str], model_name: str, dataset: str) -> Optional[str]:
    if not cache_dir:
        return None
    safe_model = str(model_name).replace("/", "_").replace(" ", "_")
    safe_ds = str(dataset).replace("/", "_").replace(" ", "_")
    return os.path.join(cache_dir, "outputs", f"{safe_model}__{safe_ds}.npz")


def save_outputs_npz(path: str, outputs: Mapping[str, np.ndarray]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in outputs.items()})
    return path


def load_outputs_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def collect_outputs_cached(
    model: Any,
    loader: Iterable,
    model_name: str,
    dataset_name: str,
    cache_dir: Optional[str] = None,
    device: Optional[Any] = None,
    overwrite: bool = False,
    max_batches: Optional[int] = None,
    desc: str = "",
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Cached version of :func:`collect_outputs` (disk reuse across stages)."""
    path = _cache_file(cache_dir, model_name, dataset_name)
    if path and os.path.exists(path) and not overwrite:
        try:
            cached = load_outputs_npz(path)
            if "logits" in cached and "targets" in cached:
                logger.debug("cache hit %s", path)
                return cached
        except Exception as exc:  # pragma: no cover - corrupt cache
            logger.warning("failed to read cache %s (%s); recomputing", path, exc)
    outputs = collect_outputs(model, loader, device=device, max_batches=max_batches, desc=desc, **kwargs)
    if path:
        try:
            save_outputs_npz(path, outputs)
        except Exception as exc:  # pragma: no cover
            logger.warning("failed to write cache %s (%s)", path, exc)
    return outputs


# ---------------------------------------------------------------------------
# Single model/dataset evaluation
# ---------------------------------------------------------------------------


def evaluate_model_on_dataset(
    logits: Any,
    targets: Any,
    hierarchy: Optional[WordNetHierarchy] = None,
    lca_matrix: Optional[Any] = None,
    metric: Optional[Any] = None,
    name: str = "",
    dataset: str = "",
    temperature: float = 1.0,
    compute_top5: bool = True,
    compute_lca: bool = True,
    compute_elca: bool = True,
    mode: str = "information",
    num_classes: Optional[int] = None,
) -> ModelMetrics:
    """Compute Top1/Top5/LCA/ELCA for one model on one dataset.

    Thin wrapper around :func:`src.metrics.lca_metric.evaluate_model_outputs`
    that keeps the paper's measurement bundle in one place.
    """
    return evaluate_model_outputs(
        logits,
        targets,
        hierarchy=hierarchy,
        lca_matrix=lca_matrix,
        mode=mode,
        num_classes=num_classes,
        temperature=temperature,
        name=name,
        dataset=dataset,
        compute_top5=compute_top5,
        compute_lca=compute_lca,
        compute_elca=compute_elca,
        metric=metric,
    )


def evaluate_model(
    model: Any,
    name: str,
    id_loader: Optional[Iterable] = None,
    ood_loaders: Optional[Mapping[str, Iterable]] = None,
    hierarchy: Optional[WordNetHierarchy] = None,
    lca_matrix: Optional[Any] = None,
    cache_dir: Optional[str] = None,
    device: Optional[Any] = None,
    temperature: float = 1.0,
    max_batches: Optional[int] = None,
    overwrite_cache: bool = False,
    family: Optional[str] = None,
    metric: Optional[Any] = None,
    num_classes: Optional[int] = None,
    id_outputs: Optional[Mapping[str, np.ndarray]] = None,
) -> ModelRecord:
    """Evaluate one model on ID and every available OOD dataset.

    ``id_outputs`` may be supplied to skip the ID forward pass (e.g. when the
    features were already needed for the latent-hierarchy stage).
    """
    started = time.time()
    record = ModelRecord(name=name, family=family or _infer_family(name))

    if id_loader is not None or id_outputs is not None:
        outputs = (
            dict(id_outputs)
            if id_outputs is not None
            else collect_outputs_cached(
                model,
                id_loader,  # type: ignore[arg-type]
                name,
                "ID",
                cache_dir=cache_dir,
                device=device,
                overwrite=overwrite_cache,
                max_batches=max_batches,
                desc=f"{name} ID",
            )
        )
        fdim = outputs.get("features")
        record.feature_dim = int(fdim.shape[1]) if fdim is not None and fdim.ndim == 2 else None
        m = evaluate_model_on_dataset(
            outputs["logits"],
            outputs["targets"],
            hierarchy=hierarchy,
            lca_matrix=lca_matrix,
            metric=metric,
            name=name,
            dataset="ID",
            temperature=temperature,
            num_classes=num_classes,
        )
        record.id_top1, record.id_top5, record.id_lca, record.id_elca = (
            m.top1,
            m.top5,
            m.lca,
            m.elca,
        )
        record.n_id = int(m.num_samples)

    for ds_name, loader in (ood_loaders or {}).items():
        try:
            outputs = collect_outputs_cached(
                model,
                loader,
                name,
                ds_name,
                cache_dir=cache_dir,
                device=device,
                overwrite=overwrite_cache,
                max_batches=max_batches,
                desc=f"{name} {ds_name}",
            )
            m = evaluate_model_on_dataset(
                outputs["logits"],
                outputs["targets"],
                hierarchy=hierarchy,
                lca_matrix=lca_matrix,
                metric=metric,
                name=name,
                dataset=ds_name,
                temperature=temperature,
                num_classes=num_classes,
            )
            record.ood[ds_name] = {
                "top1": m.top1,
                "top5": m.top5,
                "lca": m.lca,
                "lca_all": m.lca_all,
                "elca": m.elca,
                "n": m.num_samples,
            }
        except Exception as exc:  # keep the zoo sweep alive
            logger.warning("model %s on %s failed: %s", name, ds_name, exc)
            record.ood[ds_name] = {
                "top1": float("nan"),
                "top5": float("nan"),
                "lca": float("nan"),
                "lca_all": float("nan"),
                "elca": float("nan"),
                "n": 0,
            }

    record.runtime = time.time() - started
    return record


def _infer_family(name: str) -> str:
    lname = str(name).lower()
    for token in ("clip", "vlm", "blip", "albef", "slip", "openclip", "coca", "siglip"):
        if token in lname:
            return "vlm"
    return "vm"


# ---------------------------------------------------------------------------
# Zoo-level evaluation
# ---------------------------------------------------------------------------


def evaluate_zoo(
    models: Mapping[str, Any],
    id_loader: Optional[Iterable] = None,
    ood_loaders: Optional[Mapping[str, Iterable]] = None,
    hierarchy: Optional[WordNetHierarchy] = None,
    lca_matrix: Optional[Any] = None,
    config: Optional[EvaluationConfig] = None,
    show_progress: bool = True,
) -> Dict[str, ModelRecord]:
    """Evaluate a whole model zoo (36 VMs + 39 VLMs for the paper).

    Returns ``{model_name: ModelRecord}``; failures are recorded with
    ``record.error`` set instead of raising, so a long sweep is resumable.
    """
    config = config or EvaluationConfig()
    hierarchy = hierarchy or build_hierarchy()
    if lca_matrix is None:
        lca_matrix = build_lca_matrix(hierarchy)

    names = list(models.keys())
    if show_progress:
        try:
            from tqdm.auto import tqdm

            names = list(tqdm(names, desc="evaluating models"))
        except Exception:  # pragma: no cover
            pass

    records: Dict[str, ModelRecord] = {}
    for name in names:
        model = models[name]
        try:
            records[name] = evaluate_model(
                model,
                name,
                id_loader=id_loader,
                ood_loaders=ood_loaders,
                hierarchy=hierarchy,
                lca_matrix=lca_matrix,
                cache_dir=config.cache_dir,
                device=config.device,
                temperature=config.temperature,
                max_batches=config.max_batches,
                overwrite_cache=config.overwrite_cache,
                metric=None,
                num_classes=hierarchy.num_classes,
            )
            logger.info(
                "%-28s ID Top1 %.4f  LCA %.3f",
                name,
                records[name].id_top1,
                records[name].id_lca,
            )
        except Exception as exc:
            logger.error("evaluation of %s failed: %s", name, exc)
            records[name] = ModelRecord(name=name, family=_infer_family(name), error=str(exc))
    return records


def evaluate_zoo_from_cache(
    model_names: Sequence[str],
    cache_dir: str,
    hierarchy: Optional[WordNetHierarchy] = None,
    lca_matrix: Optional[Any] = None,
    ood_names: Optional[Sequence[str]] = None,
    ids: Optional[Mapping[str, int]] = None,
    families: Optional[Mapping[str, str]] = None,
    temperature: float = 1.0,
) -> Dict[str, ModelRecord]:
    """Rebuild the score table purely from cached ``.npz`` outputs.

    This is what the reproduction scripts call after features have been cached
    once: it avoids any model forward pass and lets the correlation / latent
    hierarchy / alignment stages share the same measured accuracies and LCA.
    """
    hierarchy = hierarchy or build_hierarchy()
    if lca_matrix is None:
        lca_matrix = build_lca_matrix(hierarchy)
    datasets = list(ids.keys()) if ids is not None else ["ID"] + list(
        ood_names or DEFAULT_OOD_DATASETS
    )

    records: Dict[str, ModelRecord] = {}
    for name in model_names:
        record = ModelRecord(
            name=name, family=(families or {}).get(name, _infer_family(name))
        )
        any_output = False
        for ds in datasets:
            path = _cache_file(cache_dir, name, ds)
            if not path or not os.path.exists(path):
                continue
            try:
                outputs = load_outputs_npz(path)
            except Exception as exc:  # pragma: no cover
                logger.warning("cannot read %s: %s", path, exc)
                continue
            m = evaluate_model_on_dataset(
                outputs["logits"],
                outputs["targets"],
                hierarchy=hierarchy,
                lca_matrix=lca_matrix,
                name=name,
                dataset=ds,
                temperature=temperature,
                num_classes=hierarchy.num_classes,
            )
            any_output = True
            if ds == "ID":
                record.id_top1, record.id_top5 = m.top1, m.top5
                record.id_lca, record.id_elca = m.lca, m.elca
                record.n_id = int(m.num_samples)
                feats = outputs.get("features")
                if feats is not None and feats.ndim == 2:
                    record.feature_dim = int(feats.shape[1])
            else:
                record.ood[ds] = {
                    "top1": m.top1,
                    "top5": m.top5,
                    "lca": m.lca,
                    "lca_all": m.lca_all,
                    "elca": m.elca,
                    "n": m.num_samples,
                }
        if any_output:
            records[name] = record
    return records


# ---------------------------------------------------------------------------
# Tabulation / persistence
# ---------------------------------------------------------------------------


def records_to_rows(
    records: Mapping[str, ModelRecord],
    datasets: Optional[Sequence[str]] = None,
    include_lca: bool = True,
    include_elca: bool = False,
) -> List[Dict[str, Any]]:
    """Turn records into a list of flat rows (one model per row)."""
    datasets = list(datasets or DEFAULT_OOD_DATASETS)
    rows: List[Dict[str, Any]] = []
    for name, rec in records.items():
        row: Dict[str, Any] = {
            "model": name,
            "family": rec.family,
            "ID_Top1": rec.id_top1,
            "ID_Top5": rec.id_top5,
        }
        if include_lca:
            row["ID_LCA"] = rec.id_lca
        if include_elca:
            row["ID_ELCA"] = rec.id_elca
        for ds in datasets:
            tag = display_name(ds)
            row[f"{tag}_Top1"] = rec.get(ds, "top1")
            row[f"{tag}_Top5"] = rec.get(ds, "top5")
            if include_lca:
                row[f"{tag}_LCA"] = rec.get(ds, "lca")
            if include_elca:
                row[f"{tag}_ELCA"] = rec.get(ds, "elca")
        rows.append(row)
    return rows


def records_to_dataframe(
    records: Mapping[str, ModelRecord],
    datasets: Optional[Sequence[str]] = None,
    include_elca: bool = False,
) -> Any:
    """Pandas DataFrame of the score table (Table 1 layout)."""
    import pandas as pd

    return pd.DataFrame(
        records_to_rows(records, datasets=datasets, include_elca=include_elca)
    ).set_index("model")


def summary_table(
    records: Mapping[str, ModelRecord],
    datasets: Optional[Sequence[str]] = None,
    decimals: int = 3,
) -> str:
    """Pretty printed Table 1 / Table 8 style table."""
    datasets = list(datasets or DEFAULT_OOD_DATASETS)
    header = ["Model", "Family", "ID Top1", "ID LCA", "ID ELCA"]
    for ds in datasets:
        header += [f"{display_name(ds)} Top1", f"{display_name(ds)} LCA"]
    lines = [" | ".join(header), "-" * 100]
    for name, rec in records.items():
        cells = [
            name[:28],
            rec.family,
            f"{rec.id_top1:.{decimals}f}",
            f"{rec.id_lca:.{decimals}f}",
            f"{rec.id_elca:.{decimals}f}",
        ]
        for ds in datasets:
            cells += [
                f"{rec.get(ds, 'top1'):.{decimals}f}",
                f"{rec.get(ds, 'lca'):.{decimals}f}",
            ]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def save_results(path: str, records: Mapping[str, ModelRecord], **meta: Any) -> str:
    """Persist the score table as JSON (with metadata)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "meta": meta,
        "models": {name: rec.asdict() for name, rec in records.items()},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    logger.info("wrote %d model records to %s", len(records), path)
    return path


def load_results(path: str) -> Dict[str, ModelRecord]:
    """Load a score table previously written by :func:`save_results`."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    models = payload.get("models", payload)
    records: Dict[str, ModelRecord] = {}
    for name, rec in models.items():
        rec = dict(rec)
        rec.setdefault("name", name)
        known = {f for f in ModelRecord.__dataclass_fields__}  # type: ignore[attr-defined]
        records[name] = ModelRecord(**{k: v for k, v in rec.items() if k in known})
    return records


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Driver helpers
# ---------------------------------------------------------------------------


def build_hierarchy(
    csv_path: Optional[str] = None,
    allow_synthetic: bool = True,
) -> WordNetHierarchy:
    """Build (or load) the ImageNet WordNet hierarchy used for LCA."""
    return build_wordnet_hierarchy(csv_path=csv_path, allow_synthetic=allow_synthetic)


def build_evaluation_loaders(
    config: EvaluationConfig,
    hierarchy: Optional[WordNetHierarchy] = None,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Create the ID loader plus one loader per OOD dataset."""
    from ..data.imagenet import build_imagenet_dataset, build_imagenet_loader

    hierarchy = hierarchy or build_hierarchy()
    id_dataset = build_imagenet_dataset(
        root=config.dataset_root,
        resolution=config.resolution,
        max_samples=config.max_samples,
        allow_synthetic=config.allow_synthetic,
    )
    id_loader = build_imagenet_loader(
        id_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    ood_datasets = build_all_ood_datasets(
        roots=config.ood_roots or None,
        resolution=config.resolution,
        max_samples=config.max_samples,
        allow_synthetic=config.allow_synthetic,
        names=config.ood_names,
    )
    ood_loaders: Dict[str, Any] = {}
    for name, ds in ood_datasets.items():
        ood_loaders[name] = build_imagenet_loader(
            ds,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
        )
    return id_loader, ood_loaders


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: evaluate the VM zoo (and cached VLM outputs)."""
    parser = argparse.ArgumentParser(description="Evaluate models for LCA-on-the-Line")
    parser.add_argument("--config", default=None, help="YAML config (configs/config.yaml)")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out", default="outputs/model_scores.json")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--ood", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--from-cache-only", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg_dict: Dict[str, Any] = {}
    if args.config and os.path.exists(args.config):
        import yaml

        with open(args.config, "r", encoding="utf-8") as fh:
            cfg_dict = (yaml.safe_load(fh) or {}).get("evaluation", {}) or {}
    config = EvaluationConfig.from_dict(cfg_dict)
    if args.cache_dir:
        config.cache_dir = args.cache_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_samples:
        config.max_samples = args.max_samples
    if args.max_batches:
        config.max_batches = args.max_batches
    if args.device:
        config.device = args.device
    config.overwrite_cache = config.overwrite_cache or args.overwrite_cache
    config.allow_synthetic = config.allow_synthetic or args.allow_synthetic
    if args.ood:
        config.ood_names = [normalize_ood_name(o) for o in args.ood]

    hierarchy = build_hierarchy()

    if args.from_cache_only or (config.cache_dir and not args.models):
        model_names = args.models or _discover_cached_models(config.cache_dir or "")
        records = evaluate_zoo_from_cache(
            model_names,
            config.cache_dir or "",
            hierarchy=hierarchy,
            ood_names=config.ood_names,
        )
    else:
        from ..models.vm_zoo import build_vm_zoo, list_vm_names, release_vm_zoo

        names = args.models or list_vm_names()
        id_loader, ood_loaders = build_evaluation_loaders(config, hierarchy=hierarchy)
        zoo = build_vm_zoo(names=names, device=config.device, allow_failures=True)
        records = evaluate_zoo(
            zoo,
            id_loader=id_loader,
            ood_loaders=ood_loaders,
            hierarchy=hierarchy,
            config=config,
        )
        release_vm_zoo(zoo)

    print(summary_table(records, datasets=config.ood_names))
    save_results(
        args.out,
        records,
        cache_dir=config.cache_dir,
        ood=config.ood_names,
        temperature=config.temperature,
    )
    return 0


def _discover_cached_models(cache_dir: str) -> List[str]:
    folder = os.path.join(cache_dir, "outputs")
    if not os.path.isdir(folder):
        return []
    names = set()
    for fname in os.listdir(folder):
        if "__" in fname and fname.endswith(".npz"):
            names.add(fname.split("__", 1)[0])
    return sorted(names)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
