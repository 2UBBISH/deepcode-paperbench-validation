"""Classification metrics for the Robust CLIP reproduction (ImageNet).

This module implements the *metric* side of the zero-shot ImageNet
evaluation:

* clean top-1 (and top-5) accuracy,
* robust top-1 (and top-5) accuracy under an l_inf / l_2 adversarial attack,
* the resulting accuracy drop / robustness gain,
* conditioned robustness (robust-correct among clean-correct), i.e. how often
  the prediction flips away from the correct class,
* helpers used by ``robust_clip_repro/eval_imagenet.py`` to run the clean pass
  and the adversarial pass through :class:`RunningAccuracy` trackers.

The module deliberately contains **no attack code**: attacks live in
``attacks/pgd.py`` and ``attacks/apgd.py``.  Anything the Addendum does not
state (the attack budget ``eps``, ``alpha``, the number of iterations, the
number of restarts, ...) is never invented here; those values are passed
through from the evaluation harness / config and are marked with
``UNSPECIFIED_BY_ADDENDUM`` when missing.

Everything that needs PyTorch imports it lazily so that the pure-Python
accuracy bookkeeping (and the module-level self-test) works without a GPU or
even without PyTorch installed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("robust_clip_repro.metrics.classification")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"

TOP_K_DEFAULT = 5

#: canonical ImageNet dataset name (HuggingFace id used by data/imagenet.py)
IMAGENET = "ImageNet"

#: attack families supported by the evaluation harness (see attacks/pgd.py)
SUPPORTED_NORMS: Tuple[str, ...] = ("linf", "l2")

#: values the Addendum does not specify for the ImageNet robustness evaluation
EXTERNAL_DEFAULTS: Dict[str, Any] = {
    "eps": UNSPECIFIED,
    "alpha": UNSPECIFIED,
    "iterations": UNSPECIFIED,
    "restarts": UNSPECIFIED,
    "batch_size": 1,
    "seed": 0,
    "num_workers": 0,
    "image_size": 224,
    "normalization": UNSPECIFIED,
    "prompt_template": UNSPECIFIED,
}

try:  # pragma: no cover - torch is a core dependency in the full env
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Targets / pixels extraction (duck-typed samples)
# ---------------------------------------------------------------------------

_LABEL_KEYS: Tuple[str, ...] = ("label", "labels", "target", "targets", "class_id", "class_idx", "y")
_PIXEL_KEYS: Tuple[str, ...] = ("pixels", "image_tensor", "pixel_values", "image")
_LOGIT_KEYS: Tuple[str, ...] = ("logits", "logits_all", "prediction", "predictions", "scores")


def _is_tensor(obj: Any) -> bool:
    return torch is not None and isinstance(obj, torch.Tensor)


def _to_list(values: Any) -> List[Any]:
    """Best-effort conversion of a scalar / tensor / array into a python list."""
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return list(values)
    if _is_tensor(values):
        return values.detach().cpu().reshape(-1).tolist()
    if hasattr(values, "tolist"):  # numpy array / scalar
        out = values.tolist()
        return out if isinstance(out, list) else [out]
    return [values]


def extract_label(sample: Any, default: Optional[Any] = None) -> Any:
    """Extract the ImageNet class label from a sample (dict / row / object)."""
    if sample is None:
        return default
    if isinstance(sample, dict):
        for key in _LABEL_KEYS:
            if key in sample and sample[key] is not None:
                return sample[key]
        return default
    for key in _LABEL_KEYS:
        if hasattr(sample, key):
            value = getattr(sample, key)
            if value is not None:
                return value
    if isinstance(sample, (int,)) or hasattr(sample, "__index__"):
        return sample
    return default


def extract_pixels(sample: Any, default: Optional[Any] = None) -> Any:
    """Extract raw (non-normalized) pixels from a sample."""
    if sample is None:
        return default
    if isinstance(sample, dict):
        for key in _PIXEL_KEYS:
            if key in sample and sample[key] is not None:
                return sample[key]
        return default
    for key in _PIXEL_KEYS:
        if hasattr(sample, key):
            value = getattr(sample, key)
            if value is not None:
                return value
    return default


def extract_logits(output: Any) -> Any:
    """Extract class logits from a model output (tensor / dict / tuple)."""
    if output is None:
        return None
    if _is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in _LOGIT_KEYS:
            if key in output and output[key] is not None:
                return output[key]
        return None
    if isinstance(output, (list, tuple)):
        for item in output:
            logits = extract_logits(item)
            if logits is not None:
                return logits
        return None
    if hasattr(output, "logits"):
        return output.logits
    return output


# ---------------------------------------------------------------------------
# Core accuracy computations (torch-free and torch paths)
# ---------------------------------------------------------------------------

def topk_predictions(logits: Any, k: int = TOP_K_DEFAULT) -> List[List[int]]:
    """Return the top-``k`` class indices per row of ``logits`` (list of lists)."""
    rows = _logit_rows(logits)
    if rows is None:  # pure-python path
        out: List[List[int]] = []
        for row in _iter_rows(logits):
            values = [float(v) for v in _to_list(row)]
            order = sorted(range(len(values)), key=lambda i: -values[i])
            out.append(order[: max(1, int(k))])
        return out
    return rows_topk(rows, k)


def _logit_rows(logits: Any):
    """Return the batched tensor form of ``logits`` if torch is available."""
    if torch is None:
        return None
    if logits is None:
        return None
    if not _is_tensor(logits):
        try:
            logits = torch.as_tensor(logits)
        except Exception:
            return None
    if logits.dim() == 1:
        logits = logits.reshape(1, -1)
    return logits


def _iter_rows(data: Any) -> Iterable[Any]:
    if _is_tensor(data):
        tensor = data.detach().cpu()
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, -1)
        for i in range(tensor.shape[0]):
            yield tensor[i]
        return
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
        for row in data:
            yield row
        return
    yield data


def rows_topk(logits: Any, k: int = TOP_K_DEFAULT) -> List[List[int]]:
    """Top-``k`` indices using torch when available, python otherwise."""
    tensor = _logit_rows(logits)
    if tensor is None:
        return topk_predictions(logits, k) if not _is_tensor(logits) else _python_topk(logits, k)
    kk = max(1, min(int(k), int(tensor.shape[-1])))
    return tensor.topk(kk, dim=-1).indices.detach().cpu().tolist()


def _python_topk(logits: Any, k: int) -> List[List[int]]:
    out: List[List[int]] = []
    for row in _iter_rows(logits):
        values = [float(v) for v in _to_list(row)]
        order = sorted(range(len(values)), key=lambda i: -values[i])
        out.append(order[: max(1, int(k))])
    return out


def _pairwise(logits: Any, targets: Any, k: int) -> List[bool]:
    preds = topk_predictions(logits, k)
    target_list = _to_list(targets)
    if len(target_list) != len(preds):  # broadcasting a single label / malformed
        if len(target_list) == 1 and len(preds) >= 1:
            target_list = target_list * len(preds)
        else:
            raise ValueError(
                f"targets ({len(target_list)}) and logits ({len(preds)}) length mismatch"
            )
    correct: List[bool] = []
    for row_preds, target in zip(preds, target_list):
        correct.append(int(target) in [int(p) for p in row_preds])
    return correct


def topk_correct(logits: Any, targets: Any, k: int = TOP_K_DEFAULT) -> List[bool]:
    """Per-sample boolean correctness for top-``k`` accuracy."""
    return _pairwise(logits, targets, k)


def per_sample_correct(logits: Any, targets: Any) -> List[bool]:
    """Per-sample boolean top-1 correctness."""
    return _pairwise(logits, targets, 1)


def accuracy(logits: Any, targets: Any, k: int = 1) -> float:
    """Top-``k`` accuracy in ``[0, 1]`` (0.0 for an empty batch)."""
    correct = _pairwise(logits, targets, k)
    if not correct:
        return 0.0
    return float(sum(1 for c in correct if c)) / float(len(correct))


def top1_accuracy(logits: Any, targets: Any) -> float:
    return accuracy(logits, targets, 1)


def top5_accuracy(logits: Any, targets: Any) -> float:
    return accuracy(logits, targets, TOP_K_DEFAULT)


def accuracy_from_correct(correct: Sequence[bool]) -> float:
    """Accuracy given a sequence of booleans."""
    if not len(correct):
        return 0.0
    return float(sum(1 for c in correct if c)) / float(len(correct))


# ---------------------------------------------------------------------------
# Running trackers
# ---------------------------------------------------------------------------

@dataclass
class RunningAccuracy:
    """Incremental top-k accuracy tracker (clean pass / robust pass).

    ``update`` accepts either ``(logits, targets)`` or an already computed
    sequence of booleans (``update_correct``).
    """

    k: int = 1
    name: str = "accuracy"
    total: int = 0
    correct: int = 0
    per_sample: List[bool] = field(default_factory=list)

    def update(self, logits: Any, targets: Any) -> int:
        flags = _pairwise(logits, targets, self.k)
        self.update_correct(flags)
        return len(flags)

    def update_correct(self, flags: Iterable[bool]) -> int:
        flags = [bool(f) for f in flags]
        self.total += len(flags)
        self.correct += sum(1 for f in flags if f)
        self.per_sample.extend(flags)
        return len(flags)

    def merge(self, other: "RunningAccuracy") -> "RunningAccuracy":
        self.total += other.total
        self.correct += other.correct
        self.per_sample.extend(other.per_sample)
        return self

    def reset(self) -> None:
        self.total = 0
        self.correct = 0
        self.per_sample = []

    @property
    def accuracy(self) -> Optional[float]:
        if self.total == 0:
            return None
        return float(self.correct) / float(self.total)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "k": self.k,
            "num_samples": self.total,
            "num_correct": self.correct,
            "accuracy": self.accuracy,
        }

    def __len__(self) -> int:  # pragma: no cover - convenience
        return self.total


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

@dataclass
class ClassificationReport:
    """Clean + robust top-k accuracy for one (dataset, model, attack) setting."""

    dataset_name: str = IMAGENET
    model_name: str = "model"
    attack_name: Optional[str] = None
    norm: Optional[str] = None
    eps: Optional[float] = None
    alpha: Optional[float] = None
    iterations: Optional[int] = None
    precision: Optional[str] = None
    top_k: int = TOP_K_DEFAULT
    num_samples: int = 0
    clean_top1: Optional[float] = None
    clean_top5: Optional[float] = None
    robust_top1: Optional[float] = None
    robust_top5: Optional[float] = None
    clean_correct: Optional[List[bool]] = None
    robust_correct: Optional[List[bool]] = None
    perturbation_dtype: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- derived quantities -------------------------------------------------
    @property
    def robustness_drop(self) -> Optional[float]:
        """``clean_top1 - robust_top1`` (None when either side is unknown)."""
        if self.clean_top1 is None or self.robust_top1 is None:
            return None
        return float(self.clean_top1) - float(self.robust_top1)

    @property
    def absolute_robust_gain(self) -> Optional[float]:
        return self.robust_drop

    @property
    def num_flipped(self) -> Optional[int]:
        """Samples where the clean prediction was correct but robust is not."""
        if not self.clean_correct or not self.robust_correct:
            return None
        if len(self.clean_correct) != len(self.robust_correct):
            return None
        return sum(1 for c, r in zip(self.clean_correct, self.robust_correct) if c and not r)

    @property
    def conditioned_robust_accuracy(self) -> Optional[float]:
        """Robust accuracy restricted to samples that were clean-correct."""
        if not self.clean_correct or not self.robust_correct:
            return None
        if len(self.clean_correct) != len(self.robust_correct):
            return None
        idx = [i for i, c in enumerate(self.clean_correct) if c]
        if not idx:
            return None
        return accuracy_from_correct([self.robust_correct[i] for i in idx])

    def summary(self) -> Dict[str, Any]:
        return self.as_dict()

    def as_dict(self, include_per_sample: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "attack": self.attack_name,
            "norm": self.norm,
            "eps": self.eps,
            "alpha": self.alpha,
            "iterations": self.iterations,
            "precision": self.precision,
            "perturbation_dtype": self.perturbation_dtype,
            "num_samples": self.num_samples,
            "top_k": self.top_k,
            "clean_top1": self.clean_top1,
            "clean_top5": self.clean_top5,
            "robust_top1": self.robust_top1,
            "robust_top5": self.robust_top5,
            "robustness_drop": self.robustness_drop,
            "num_flipped": self.num_flipped,
            "conditioned_robust_accuracy": self.conditioned_robust_accuracy,
        }
        if include_per_sample:
            payload["clean_correct"] = list(self.clean_correct or [])
            payload["robust_correct"] = list(self.robust_correct or [])
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


# ---------------------------------------------------------------------------
# Evaluation helpers used by eval_imagenet.py
# ---------------------------------------------------------------------------

def _as_list(samples: Any) -> List[Any]:
    if samples is None:
        return []
    if isinstance(samples, (list, tuple)):
        return list(samples)
    if _is_tensor(samples):
        return list(samples)
    if hasattr(samples, "__len__") and hasattr(samples, "__getitem__"):
        return [samples[i] for i in range(len(samples))]
    return list(samples)


def collect_batch(samples: Any, *, device: Any = None) -> Tuple[Any, Any, List[Any]]:
    """Stack raw pixels + labels of a list of samples into ``(pixels, labels, raw)``.

    Returns python objects when torch is unavailable.  Pixels are stacked in
    their **raw (non-normalized)** space -- the space in which the Addendum
    requires the l_inf ball to be computed.
    """
    items = _as_list(samples)
    pixels_list: List[Any] = []
    labels: List[Any] = []
    for sample in items:
        px = extract_pixels(sample)
        if px is None:
            raise KeyError("sample has no pixel field among %s" % (_PIXEL_KEYS,))
        pixels_list.append(px)
        labels.append(extract_label(sample))

    if torch is not None and all(_is_tensor(p) for p in pixels_list) and pixels_list:
        pixels = torch.stack([p.detach() for p in pixels_list], dim=0)
        if device is not None:
            pixels = pixels.to(device)
        if all(l is not None for l in labels):
            targets = torch.as_tensor(labels, dtype=torch.long, device=pixels.device)
        else:
            targets = labels
        return pixels, targets, items

    targets = torch.as_tensor(labels, dtype=torch.long) if (torch is not None and all(l is not None for l in labels)) else labels
    return pixels_list, targets, items


def make_logits_fn(predict_fn: Callable[..., Any]) -> Callable[[Any], Any]:
    """Wrap ``predict_fn`` so it always returns bare class logits."""

    def logits_fn(pixels: Any) -> Any:
        return extract_logits(predict_fn(pixels))

    return logits_fn


def run_clean_pass(
    predict_fn: Callable[..., Any],
    samples: Any,
    *,
    batch_size: int = 1,
    top_k: int = TOP_K_DEFAULT,
    device: Any = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compute clean top-1/top-5 accuracy over ``samples``.

    ``predict_fn(pixels) -> logits`` must accept a batch of raw pixels and
    return class logits (dict / tensor / object with ``.logits`` are all
    accepted through :func:`extract_logits`).
    """
    items = _as_list(samples)
    top1 = RunningAccuracy(k=1, name="clean_top1")
    topk = RunningAccuracy(k=top_k, name="clean_top%d" % top_k)
    for start in range(0, len(items), max(1, int(batch_size))):
        batch = items[start : start + max(1, int(batch_size))]
        pixels, targets, _ = collect_batch(batch, device=device)
        logits = extract_logits(predict_fn(pixels))
        top1.update(logits, targets)
        if top_k and top_k > 1:
            topk.update(logits, targets)
        if verbose:
            LOGGER.info("clean %d/%d", min(start + len(batch), len(items)), len(items))
    return {
        "num_samples": len(items),
        "clean_top1": top1.accuracy,
        "clean_top%d" % top_k: topk.accuracy if top_k and top_k > 1 else top1.accuracy,
        "clean_correct": list(top1.per_sample),
        "clean_correct_top%d" % top_k: list(topk.per_sample) if top_k and top_k > 1 else list(top1.per_sample),
    }


def run_robust_pass(
    predict_fn: Callable[..., Any],
    attack: Any,
    samples: Any,
    *,
    batch_size: int = 1,
    top_k: int = TOP_K_DEFAULT,
    device: Any = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the adversarial pass with a PGD/APGD attack object.

    ``attack`` must follow the interface shared by
    :class:`robust_clip_repro.attacks.pgd.PGDLinfAttack` and
    :class:`robust_clip_repro.attacks.apgd.APGDAttack`: it exposes
    ``attack_untargeted(pixels, loss_fn, labels=...)`` returning raw-pixel
    adversarial examples (or an integer-coded perturbation, which is added to
    the raw pixels here).

    The attack budget (``eps``/``alpha``/``iterations``) lives on the attack
    object -- nothing is invented in this metric module.
    """
    items = _as_list(samples)
    top1 = RunningAccuracy(k=1, name="robust_top1")
    topk = RunningAccuracy(k=top_k, name="robust_top%d" % top_k)
    perturbation_dtype: Optional[str] = None

    for start in range(0, len(items), max(1, int(batch_size))):
        batch = items[start : start + max(1, int(batch_size))]
        pixels, targets, _ = collect_batch(batch, device=device)
        logits_fn = make_logits_fn(predict_fn)

        def loss_fn(x: Any, _logits_fn: Callable[..., Any] = logits_fn) -> Any:
            logits = _logits_fn(x)
            return _cross_entropy(logits, targets)

        adversarial = attack.attack_untargeted(pixels, loss_fn, labels=targets)
        adversarial = _to_pixels(adversarial, attack, pixels)
        perturbation_dtype = _dtype_name(_compute_delta(attack, pixels, adversarial))
        logits = extract_logits(predict_fn(adversarial))
        top1.update(logits, targets)
        if top_k and top_k > 1:
            topk.update(logits, targets)
        if verbose:
            LOGGER.info("robust %d/%d", min(start + len(batch), len(items)), len(items))

    return {
        "num_samples": len(items),
        "robust_top1": top1.accuracy,
        "robust_top%d" % top_k: topk.accuracy if top_k and top_k > 1 else top1.accuracy,
        "robust_correct": list(top1.per_sample),
        "robust_correct_top%d" % top_k: list(topk.per_sample) if top_k and top_k > 1 else list(top1.per_sample),
        "perturbation_dtype": perturbation_dtype,
    }


def _dtype_name(tensor: Any) -> Optional[str]:
    if tensor is None:
        return None
    dtype = getattr(tensor, "dtype", None)
    return None if dtype is None else str(dtype).replace("torch.", "")


def _compute_delta(attack: Any, pixels: Any, adversarial: Any) -> Any:
    """Recover the (integer-coded or float) perturbation actually applied."""
    for attr in ("last_perturbation", "last_delta", "delta"):
        delta = getattr(attack, attr, None)
        if delta is not None:
            return delta
    try:
        return adversarial - pixels
    except Exception:  # pragma: no cover - defensive
        return None


def _to_pixels(adversarial: Any, attack: Any, pixels: Any) -> Any:
    """Normalize the attack output into raw adversarial pixels."""
    if adversarial is None:
        return pixels
    dtype = getattr(adversarial, "dtype", None)
    if dtype is not None and "int" in str(dtype) and hasattr(attack, "adversarial_examples"):
        try:
            return attack.adversarial_examples(pixels, adversarial)
        except Exception:  # pragma: no cover - defensive
            return adversarial
    return adversarial


def _cross_entropy(logits: Any, targets: Any) -> Any:
    if torch is not None:
        return torch.nn.functional.cross_entropy(logits, targets)
    raise RuntimeError("torch is required for the robust classification pass")


def evaluate_classification(
    predict_fn: Callable[..., Any],
    samples: Any,
    *,
    attack: Any = None,
    dataset_name: str = IMAGENET,
    model_name: str = "model",
    attack_name: Optional[str] = None,
    norm: Optional[str] = None,
    eps: Optional[float] = None,
    alpha: Optional[float] = None,
    iterations: Optional[int] = None,
    precision: Optional[str] = None,
    batch_size: int = 1,
    top_k: int = TOP_K_DEFAULT,
    device: Any = None,
    verbose: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> ClassificationReport:
    """Clean (+ optional robust) top-1/top-5 evaluation returning a report."""
    clean = run_clean_pass(
        predict_fn, samples, batch_size=batch_size, top_k=top_k, device=device, verbose=verbose
    )
    report = ClassificationReport(
        dataset_name=dataset_name,
        model_name=model_name,
        attack_name=attack_name,
        norm=norm,
        eps=eps,
        alpha=alpha,
        iterations=iterations,
        precision=precision,
        top_k=top_k,
        num_samples=clean["num_samples"],
        clean_top1=clean["clean_top1"],
        clean_top5=clean["clean_top5"],
        clean_correct=clean["clean_correct"],
        extra=dict(extra or {}),
    )
    if attack is not None:
        robust = run_robust_pass(
            predict_fn,
            attack,
            samples,
            batch_size=batch_size,
            top_k=top_k,
            device=device,
            verbose=verbose,
        )
        report.robust_top1 = robust["robust_top1"]
        report.robust_top5 = robust["robust_top5"]
        report.robust_correct = robust["robust_correct"]
        report.perturbation_dtype = robust["perturbation_dtype"]
        report.norm = norm if norm is not None else getattr(attack, "norm", None)
        report.eps = eps if eps is not None else getattr(attack, "eps", None)
        report.alpha = alpha if alpha is not None else getattr(attack, "alpha", None)
        report.iterations = iterations if iterations is not None else getattr(attack, "iterations", None)
        report.precision = precision if precision is not None else getattr(attack, "precision", None)
    return report


# ---------------------------------------------------------------------------
# Comparison tables (paper trends: Robust CLIP vs vanilla CLIP)
# ---------------------------------------------------------------------------

@dataclass
class ComparisonRow:
    """One row of the clean/robust comparison table."""

    method: str
    norm: Optional[str] = None
    eps: Optional[float] = None
    clean_top1: Optional[float] = None
    robust_top1: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "norm": self.norm,
            "eps": self.eps,
            "clean_top1": self.clean_top1,
            "robust_top1": self.robust_top1,
            "drop": None
            if (self.clean_top1 is None or self.robust_top1 is None)
            else float(self.clean_top1) - float(self.robust_top1),
        }


def build_comparison_table(reports: Iterable[ClassificationReport]) -> List[Dict[str, Any]]:
    """Tabulate several reports (e.g. vanilla vs Robust CLIP, l_inf vs l_2)."""
    return [
        ComparisonRow(
            method=r.model_name,
            norm=r.norm,
            eps=r.eps,
            clean_top1=r.clean_top1,
            robust_top1=r.robust_top1,
        ).as_dict()
        for r in reports
    ]


def format_table(rows: Sequence[Dict[str, Any]], *, percent: bool = True) -> str:
    """Render comparison rows as a plain-text table (for the README/logs)."""
    header = ("method", "norm", "eps", "clean_top1", "robust_top1", "drop")
    lines = ["\t".join(header)]
    for row in rows:
        cells: List[str] = []
        for key in header:
            value = row.get(key)
            if percent and isinstance(value, float) and key != "eps":
                cells.append("%.2f" % (100.0 * value))
            elif value is None:
                cells.append("-")
            elif isinstance(value, float):
                cells.append("%.4f" % value)
            else:
                cells.append(str(value))
        lines.append("\t".join(cells))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI / self-test
# ---------------------------------------------------------------------------

def _toy_accuracy(base: float = 1.0, num_classes: int = 10, num_samples: int = 8):
    """Deterministic (logits, targets) pair with controlled top-1 accuracy."""
    if torch is None:
        logits: List[List[float]] = []
        targets: List[int] = []
        for i in range(num_samples):
            row = [0.0] * num_classes
            target = i % num_classes
            right = (i / max(1, num_samples)) < base
            row[target if right else (target + 1) % num_classes] = 1.0
            logits.append(row)
            targets.append(target)
        return logits, targets
    g = torch.Generator().manual_seed(0)
    targets = torch.arange(num_samples) % num_classes
    logits = torch.randn(num_samples, num_classes, generator=g)
    num_correct = int(round(base * num_samples))
    for i in range(num_samples):
        if i < num_correct:
            logits[i, targets[i]] = logits[i].max().item() + 1.0
    return logits, targets


def _self_test(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks for the top-k machinery, tracker and report."""
    logits, targets = _toy_accuracy(1.0, num_classes=10, num_samples=8)
    assert top1_accuracy(logits, targets) == 1.0, "perfect accuracy expected"
    assert top5_accuracy(logits, targets) == 1.0, "top-5 >= top-1 expected"

    logits2, targets2 = _toy_accuracy(0.5, num_classes=10, num_samples=8)
    acc = top1_accuracy(logits2, targets2)
    assert 0.0 < acc < 1.0, "partial accuracy expected, got %s" % acc
    assert top5_accuracy(logits2, targets2) >= acc, "top-5 must dominate top-1"

    empty = RunningAccuracy(k=1, name="empty")
    assert empty.accuracy is None

    tracker = RunningAccuracy(k=1, name="clean_top1")
    tracker.update(logits2, targets2)
    partial = RunningAccuracy(k=1, name="part")
    partial.update(logits2[:4], targets2[:4])
    tracker.merge(partial)
    assert tracker.total == len(targets2) + 4

    tracker.reset()
    assert tracker.total == 0 and tracker.accuracy is None

    report = ClassificationReport(
        model_name="robust_clip",
        norm="linf",
        eps=4 / 255,
        clean_top1=0.75,
        robust_top1=0.55,
        clean_correct=[True, True, True, False],
        robust_correct=[True, False, False, False],
    )
    payload = report.as_dict()
    assert abs(payload["robustness_drop"] - 0.20) < 1e-9, payload["robustness_drop"]
    assert payload["num_flipped"] == 2, payload["num_flipped"]
    assert abs(payload["conditioned_robust_accuracy"] - (1 / 3)) < 1e-9

    table = build_comparison_table([report])
    assert table and table[0]["method"] == "robust_clip"
    text = format_table(table)
    assert "clean_top1" in text and "55.00" in text, text

    # sanity: norms accepted, defaults marked unspecified
    assert "linf" in SUPPORTED_NORMS and "l2" in SUPPORTED_NORMS
    assert EXTERNAL_DEFAULTS["eps"] == UNSPECIFIED

    if verbose:
        print(json.dumps(payload, indent=2))
        print(text)
    return {"report": payload, "table": table}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classification metrics for the Robust CLIP reproduction (module self-test)."
    )
    parser.add_argument("--self-test", action="store_true", help="run offline metric checks")
    parser.add_argument("--quiet", action="store_true", help="suppress self-test output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    if args.self_test or not argv:
        result = _self_test(verbose=not args.quiet)
        if args.quiet:
            print(json.dumps(result["report"]))
        return 0
    build_arg_parser().print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
