"""FOA-I: interval-update FOA for single-sample (BS = 1) adaptation.

Paper reference
---------------
Section 4.4 ("Results on Single Sample Adaptation (Batch Size = 1)"):

    "In our FOA, the prompt is learned over a batch of test samples each time.
     This process suffers from a challenge when the batch size is limited to one,
     as it requires the computation of the mean and variance of features, which
     may not be feasible with a single sample. ... We propose a solution in the
     form of an interval update strategy, referred to as FOA-I. Specifically,
     given an ongoing stream of test data, we opt to update the prompts
     (performing CMA optimization) after encountering a pre-defined number of
     samples, denoted as I. During this interval, we temporarily store the
     relevant features of all CLS tokens or the original image until the next
     update."

Addendum:

    "Table 7 mentions 'FOA-I V1/V2' which refer to two different implementation
     strategies for interval-based FOA: V1 stores features between updates while
     V2 stores images between updates."

    "For the activation shifting moving average (Equation 9), mu_N(0) should be
     initialized using the statistics of the first batch mu_N(X_1)."

Implemented semantics
---------------------
Interval loop, for every incoming single sample x_t (t = 1, 2, ...):

  1. If a new interval starts, ask the CMA optimizer for the K candidate prompt
     vectors of this window (Eqn. 6) and snapshot the activation-shifting state
     mu_N(t-1) at the window start.
  2. Forward x_t with the *incumbent* prompt (best candidate of the previous
     window), classify after activation shifting with the live state
     (Eqn. 7-8: e_N <- e_N + gamma * (mu_N^S - mu_N(t-1))), emit the prediction,
     then update the moving average with the *un-shifted* final-layer CLS
     feature (Eqn. 9: mu_N(t) = alpha * mu_N(X_t) + (1-alpha) * mu_N(t-1),
     initialized with mu_N(X_1)).
  3. Buffer the window data -- V1 stores the all-layer CLS features obtained
     with each of the K pending candidates (so the boundary update needs no
     further forward pass), V2 stores the raw image (so the boundary update
     recomputes features for every candidate).
  4. Once I samples have accumulated, perform ONE CMA update over the window:
     score each candidate with the Eqn. (5) fitness on the window statistics
     (entropy + lambda * activation discrepancy vs. the source bank), `tell()`
     the K fitness values, and adopt the best candidate as the new incumbent.
     The buffer is then cleared and the next interval begins.

Both variants evaluate the *same* candidate set on the *same* window, hence they
share the accuracy reported for FOA-I in Table 6 and differ only in memory usage
(Table 7, which the addendum excludes from reproduction).

Nothing here ever calls ``backward()``: the frozen backbone runs under
``torch.no_grad()`` and only the CMA search distribution plus the shifting EMA
state change over time.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from the rest of the package (defensive: the module stays importable
# even when only a subset of the package is available).
# ---------------------------------------------------------------------------
try:  # package-relative
    from .foa import FOA, build_foa, unpack_batch  # type: ignore
except Exception:  # pragma: no cover - script style imports
    try:
        from src.method.foa import FOA, build_foa, unpack_batch  # type: ignore
    except Exception:
        FOA = None  # type: ignore
        build_foa = None  # type: ignore

        def unpack_batch(batch: Any) -> Tuple[Any, Optional[Any]]:
            """Minimal fallback batch unpacker (dict / tuple / tensor)."""
            if isinstance(batch, dict):
                images = None
                for key in ("image", "images", "x", "input"):
                    if key in batch:
                        images = batch[key]
                        break
                if images is None:
                    raise KeyError(f"could not find images in batch keys {list(batch)}")
                targets = None
                for key in ("label", "labels", "y", "target", "targets"):
                    if key in batch:
                        targets = batch[key]
                        break
                return images, targets
            if isinstance(batch, (tuple, list)):
                if len(batch) >= 2:
                    return batch[0], batch[1]
                return batch[0], None
            return batch, None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FOA_I_INTERVALS: Tuple[int, ...] = (4, 8, 16, 32, 64)
"""Intervals I swept in Table 6 / Table 7 of the paper."""

FOA_I_VARIANTS: Tuple[str, ...] = ("v1", "v2")
"""V1 = cache features between updates, V2 = cache images between updates."""

DEFAULT_INTERVAL: int = 4
"""Smallest interval of the paper sweep (Table 6, best accuracy 62.1%)."""

VARIANT_FEATURES: str = "v1"
VARIANT_IMAGES: str = "v2"

DEFAULT_ECE_BINS: int = 15
"""Equal-width ECE bins (documented ambiguity (d) of the plan)."""

TABLE6_REFERENCE: Dict[str, Any] = {
    "dataset": "ImageNet-C (Gaussian noise, severity 5)",
    "model": "ViT-Base (32-bit)",
    "noadapt": {"accuracy": 56.8, "ece": 7.5},
    "tent_bs64": {"accuracy": 60.3, "ece": 13.7},
    "foa_i": {
        4: {"accuracy": 62.1, "ece": 3.3},
        8: {"accuracy": 62.2, "ece": 2.6},
        16: {"accuracy": 62.1, "ece": 2.8},
        32: {"accuracy": 61.9, "ece": 2.7},
        64: {"accuracy": 61.5, "ece": 2.5},
    },
}

__all__ = [
    "FOAInterval",
    "FOAI",
    "build_foa_i",
    "run_foa_i",
    "run_interval_sweep",
    "normalize_variant",
    "print_table6",
    "FOA_I_INTERVALS",
    "FOA_I_VARIANTS",
    "DEFAULT_INTERVAL",
    "TABLE6_REFERENCE",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Dotted-path lookup working with dicts, ``Config`` and attribute objects."""
    if cfg is None:
        return default
    for key in keys:
        node = cfg
        found = True
        for part in str(key).split("."):
            if node is None:
                found = False
                break
            if isinstance(node, dict):
                if part in node:
                    node = node[part]
                else:
                    found = False
                    break
            else:
                node = getattr(node, part, None)
                if node is None:
                    found = False
                    break
        if found and node is not None:
            return node
    return default


def normalize_variant(variant: Any) -> str:
    """Map user spellings onto the canonical ``"v1"`` / ``"v2"`` variant names."""
    if variant is None:
        return VARIANT_FEATURES
    name = str(variant).strip().lower().replace("-", "").replace("_", "")
    if name in ("v1", "1", "feature", "features", "feat", "cls", "clsfeatures"):
        return VARIANT_FEATURES
    if name in ("v2", "2", "image", "images", "img", "raw"):
        return VARIANT_IMAGES
    raise ValueError(f"unknown FOA-I variant {variant!r}; expected one of {FOA_I_VARIANTS}")


def _resolve_interval(cfg: Any, interval: Optional[int] = None) -> int:
    if interval is None:
        interval = _cfg_get(cfg, "foa_i.interval", "interval", "foai.interval", default=None)
    if interval is None:
        interval = DEFAULT_INTERVAL
    interval = int(interval)
    if interval < 1:
        raise ValueError(f"interval I must be >= 1, got {interval}")
    return interval


def _resolve_variant(cfg: Any, variant: Optional[str] = None) -> str:
    if variant is None:
        variant = _cfg_get(cfg, "foa_i.variant", "variant", "foai.variant", default=None)
    return normalize_variant(variant)


def _to_image_tensor(image: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    """Normalize a single image (or a 1-sample batch) to ``[B, 3, H, W]`` float32."""
    if isinstance(image, torch.Tensor):
        tensor = image
    elif isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image)
    else:  # PIL image or list of tensors
        if isinstance(image, (list, tuple)) and image and isinstance(image[0], torch.Tensor):
            tensor = torch.stack(list(image), dim=0)
        else:  # pragma: no cover - relies on torchvision being available upstream
            tensor = torch.as_tensor(np.asarray(image))
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 2:  # pragma: no cover - flattened CHW
        tensor = tensor.view(1, 1, tensor.shape[0], tensor.shape[1])
    tensor = tensor.to(dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _extract_output(out: Any) -> Dict[str, Any]:
    """Normalize the backbone output into ``{'cls_features', 'logits', 'final_cls'}``."""
    if isinstance(out, dict):
        cls_features = None
        for key in ("cls_features", "cls_feats", "features", "all_cls"):
            if out.get(key) is not None:
                cls_features = out[key]
                break
        logits = out.get("logits")
        final_cls = None
        for key in ("final_cls", "final", "cls"):
            if out.get(key) is not None:
                final_cls = out[key]
                break
    elif isinstance(out, (tuple, list)):
        cls_features = out[0] if len(out) > 0 else None
        logits = out[1] if len(out) > 1 else None
        final_cls = out[2] if len(out) > 2 else None
    else:
        cls_features, logits, final_cls = None, out, None

    if cls_features is not None and not isinstance(cls_features, (list, tuple)):
        cls_features = [cls_features]
    if final_cls is None and cls_features:
        final_cls = cls_features[-1]
    if not isinstance(cls_features, (list, tuple)):
        cls_features = []
    return {
        "cls_features": list(cls_features),
        "logits": logits,
        "final_cls": final_cls,
    }


def _stack_features(per_sample: Sequence[Sequence[torch.Tensor]]) -> List[torch.Tensor]:
    """Stack per-sample ``[1, d]`` per-layer features into per-layer ``[B, d]``."""
    per_sample = [s for s in per_sample if s is not None and len(s) > 0]
    if not per_sample:
        return []
    num_layers = min(len(s) for s in per_sample)
    stacked: List[torch.Tensor] = []
    for i in range(num_layers):
        parts = []
        for sample_features in per_sample:
            feats = sample_features[i]
            if feats.dim() == 1:
                feats = feats.unsqueeze(0)
            parts.append(feats)
        stacked.append(torch.cat(parts, dim=0))
    return stacked


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)


def _accuracy_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = DEFAULT_ECE_BINS) -> Tuple[float, float]:
    """Top-1 accuracy (%) and equal-width ECE (%) from probabilities."""
    if probs.size == 0:
        return float("nan"), float("nan")
    preds = probs.argmax(axis=-1)
    acc = 100.0 * float((preds == targets).mean())
    conf = probs.max(axis=-1)
    correct = (preds == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_ids = np.digitize(conf, edges[1:-1], right=False)
    ece = 0.0
    total = len(conf)
    for b in range(int(n_bins)):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        acc_b = correct[mask].mean()
        conf_b = conf[mask].mean()
        ece += (count / total) * abs(acc_b - conf_b)
    return acc, 100.0 * float(ece)


class _IntervalAccumulator:
    """Streaming accuracy/ECE accumulator used when ``src.eval.metrics`` is absent."""

    def __init__(self, n_bins: int = DEFAULT_ECE_BINS) -> None:
        self.n_bins = int(n_bins)
        self._probs: List[np.ndarray] = []
        self._targets: List[np.ndarray] = []

    def update(self, logits: Any, targets: Any) -> Tuple[float, float]:
        with torch.no_grad():
            logits_np = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
            targets_np = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
        logits_np = np.atleast_2d(logits_np.astype(np.float64))
        targets_np = np.atleast_1d(targets_np).astype(np.int64)
        probs = _softmax_np(logits_np)
        self._probs.append(probs)
        self._targets.append(targets_np)
        return _accuracy_ece(probs, targets_np, self.n_bins)

    def compute(self) -> Dict[str, float]:
        if not self._probs:
            return {"accuracy": float("nan"), "ece": float("nan"), "num_samples": 0}
        probs = np.concatenate(self._probs, axis=0)
        targets = np.concatenate(self._targets, axis=0)
        acc, ece = _accuracy_ece(probs, targets, self.n_bins)
        return {"accuracy": acc, "ece": ece, "num_samples": int(len(targets))}

    def reset(self) -> None:
        self._probs.clear()
        self._targets.clear()


def _build_accumulator(cfg: Any = None) -> Any:
    """Prefer the project's ``MetricAccumulator``; otherwise use the local fallback."""
    n_bins = int(_cfg_get(cfg, "eval.ece_bins", default=DEFAULT_ECE_BINS) or DEFAULT_ECE_BINS)
    try:
        from ..eval.metrics import MetricAccumulator  # type: ignore

        try:
            return MetricAccumulator(n_bins=n_bins)
        except TypeError:  # pragma: no cover - alternate signature
            return MetricAccumulator()
    except Exception:
        try:
            from src.eval.metrics import MetricAccumulator  # type: ignore

            try:
                return MetricAccumulator(n_bins=n_bins)
            except TypeError:  # pragma: no cover
                return MetricAccumulator()
        except Exception:
            return _IntervalAccumulator(n_bins=n_bins)


# ---------------------------------------------------------------------------
# FOA-I
# ---------------------------------------------------------------------------
@dataclass
class IntervalStats:
    """Bookkeeping for the interval loop."""

    samples_seen: int = 0
    updates: int = 0
    skipped_candidates: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "samples_seen": self.samples_seen,
            "updates": self.updates,
            "skipped_candidates": self.skipped_candidates,
        }


class FOAInterval:
    """Interval-update FOA (FOA-I) for batch-size-1 streams.

    Parameters
    ----------
    foa
        A (constructed) :class:`~src.method.foa.FOA` instance providing the frozen
        backbone, the learnable prompt, the CMA optimizer, the Eqn. (5) fitness
        and the activation shifter.
    interval
        Number of samples buffered before one CMA update (``I``, Table 6).
    variant
        ``"v1"`` caches the all-layer CLS features of the window (extracted with
        the K pending candidate prompts, so the boundary update performs no extra
        forward pass); ``"v2"`` caches the raw images and recomputes features for
        every candidate at the boundary. Both evaluate the same candidate set on
        the same window, so their accuracy coincides (Table 6) while memory use
        differs (Table 7).
    update_partial_buffer
        Run a final CMA update on a leftover partial window when the stream ends
        (keeps the search distribution fresh; does not change emitted
        predictions).
    shift_candidates
        Apply the window-level activation shift to the candidate final-layer CLS
        features before the head when scoring Eqn. (5) (mirrors the per-batch
        behaviour of the main FOA loop; set ``False`` for the Table 5
        "entropy + discrepancy without shifting" variant).
    """

    def __init__(
        self,
        foa: Any,
        interval: int = DEFAULT_INTERVAL,
        variant: str = VARIANT_FEATURES,
        *,
        update_partial_buffer: bool = True,
        shift_candidates: bool = True,
        device: Optional[Any] = None,
        population_size: Optional[int] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        if foa is None:
            raise ValueError("FOAInterval requires a constructed FOA instance (foa=...)")
        self.foa = foa
        self.interval = _resolve_interval(None, interval)
        self.variant = normalize_variant(variant)
        self.update_partial_buffer = bool(update_partial_buffer)
        self.shift_candidates = bool(shift_candidates)
        self.verbose = bool(verbose)

        # --- components of the underlying FOA runner -------------------------
        self.model = getattr(foa, "model", None)
        self.prompt = getattr(foa, "prompt", None)
        self.optimizer = getattr(foa, "optimizer", None)
        self.fitness = getattr(foa, "fitness", None)
        self.shifter = getattr(foa, "shifter", None)
        self.source_stats = getattr(foa, "source_stats", None)

        self.device = (
            device
            if device is not None
            else getattr(foa, "device", None)
        )
        if self.device is None:
            try:
                self.device = next(self.model.parameters()).device
            except Exception:
                self.device = torch.device("cpu")

        self.population_size = int(
            population_size
            or getattr(self.optimizer, "population_size", 0)
            or _cfg_get(getattr(foa, "cfg", None), "cma.population_size", default=28)
        )
        if seed is not None:
            np.random.seed(int(seed))

        # dimension of the flattened prompt search space (d * N_p)
        self.prompt_dim = int(
            getattr(self.prompt, "prompt_dim", 0)
            or _cfg_get(getattr(foa, "cfg", None), "model.embed_dim", default=0)
            or 768 * int(_cfg_get(getattr(foa, "cfg", None), "prompt.num_prompts", default=3) or 3)
        )

        # --- online state -----------------------------------------------------
        self.stats = IntervalStats()
        self.incumbent = self._initial_incumbent()
        self._buffer: List[Dict[str, Any]] = []
        self._pending_candidates: Optional[np.ndarray] = None
        self._window_shift_state: Optional[Any] = None
        self._last_values: Optional[List[float]] = None
        self._last_best: Optional[int] = None
        self._window_index = 0

    # -- construction helpers ------------------------------------------------
    def _initial_incumbent(self) -> np.ndarray:
        """Initial incumbent prompt: the prompt module's own uniform init."""
        vector = None
        if self.prompt is not None and hasattr(self.prompt, "get_prompt"):
            try:
                vector = self.prompt.get_prompt()
            except Exception:
                vector = None
        if vector is None:
            vector = np.zeros(max(self.prompt_dim, 1), dtype=np.float64)
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()
        return np.asarray(vector, dtype=np.float64).reshape(-1).copy()

    # -- prompt / shifting primitives ---------------------------------------
    def _set_prompt(self, vector: Optional[np.ndarray]) -> None:
        if self.prompt is None or vector is None:
            return
        if int(getattr(self.prompt, "num_prompts", 1) or 0) == 0:
            return
        tensor = torch.as_tensor(np.asarray(vector).reshape(-1), dtype=torch.float32)
        if self.device is not None:
            tensor = tensor.to(self.device)
        try:
            self.prompt.set_prompt(tensor)
        except Exception:  # pragma: no cover - tolerate signature variations
            self.prompt.set_prompt(np.asarray(vector).reshape(-1))

    def _prompt_tensor(self) -> Optional[torch.Tensor]:
        if self.prompt is None:
            return None
        if int(getattr(self.prompt, "num_prompts", 1) or 0) == 0:
            return None
        try:
            return self.prompt.as_tensor()
        except Exception:  # pragma: no cover
            return None

    def _head(self, final_cls: torch.Tensor) -> torch.Tensor:
        if self.model is None or not hasattr(self.model, "head"):
            raise AttributeError("model does not expose a classification head")
        return self.model.head(final_cls)

    def _snapshot(self) -> Optional[Any]:
        """Deep copy of the activation-shifting state (``mu_N(t-1)``)."""
        if self.shifter is None:
            return None
        for name in ("_snapshot_shift_state",):
            fn = getattr(self.foa, name, None)
            if callable(fn):
                try:
                    return copy.deepcopy(fn())
                except Exception:
                    pass
        if hasattr(self.shifter, "state_dict"):
            try:
                return copy.deepcopy(self.shifter.state_dict())
            except Exception:
                return None
        for attr in ("mu_test", "mu_N", "running_mean"):
            if hasattr(self.shifter, attr):
                return {"attr": attr, "value": copy.deepcopy(getattr(self.shifter, attr))}
        return None

    def _restore(self, state: Optional[Any]) -> None:
        if self.shifter is None or state is None:
            return
        fn = getattr(self.foa, "_restore_shift_state", None)
        if callable(fn):
            try:
                fn(state)
                return
            except Exception:
                pass
        if hasattr(self.shifter, "load_state_dict") and isinstance(state, dict) and "attr" not in state:
            try:
                self.shifter.load_state_dict(state)
                return
            except Exception:
                pass
        if isinstance(state, dict) and state.get("attr"):
            try:
                setattr(self.shifter, state["attr"], state["value"])
            except Exception:  # pragma: no cover
                pass

    def _shift(self, final_cls: torch.Tensor) -> torch.Tensor:
        """Eqn. (7)-(8): ``e_N + gamma * (mu_N^S - mu_N(t-1))`` (no state update)."""
        if self.shifter is None:
            return final_cls
        fn = getattr(self.foa, "shift_features", None)
        if callable(fn):
            try:
                return fn(final_cls)
            except Exception:
                pass
        fn = getattr(self.shifter, "shift", None)
        if callable(fn):
            try:
                return fn(final_cls)
            except Exception:
                pass
        # manual fallback
        mu_prev = None
        for name in ("current_mean",):
            fn = getattr(self.shifter, name, None)
            if callable(fn):
                try:
                    mu_prev = fn()
                    break
                except Exception:
                    mu_prev = None
        if mu_prev is None:
            for attr in ("mu_test", "mu_N", "running_mean"):
                if hasattr(self.shifter, attr):
                    mu_prev = getattr(self.shifter, attr)
                    break
        mu_src = None
        for attr in ("mu_final", "mu_N"):
            if self.source_stats is not None and hasattr(self.source_stats, attr):
                mu_src = getattr(self.source_stats, attr)
                break
        if mu_prev is None or mu_src is None:
            return final_cls
        gamma = float(getattr(self.shifter, "gamma", 1.0) or 1.0)
        return final_cls + gamma * (mu_src.to(final_cls.device, final_cls.dtype) - mu_prev.to(final_cls.device, final_cls.dtype))

    def _update_shift(self, final_cls: torch.Tensor) -> None:
        """Eqn. (9): EMA update from the *un-shifted* final-layer CLS feature."""
        if self.shifter is None:
            return
        fn = getattr(self.foa, "update_test_mean", None)
        if callable(fn):
            try:
                fn(final_cls)
                return
            except Exception:
                pass
        fn = getattr(self.shifter, "update", None)
        if callable(fn):
            try:
                fn(final_cls)
                return
            except Exception:
                pass
        # manual fallback
        alpha = float(getattr(self.shifter, "alpha", 0.1) or 0.1)
        batch_mean = final_cls.detach().mean(dim=0)
        previous = None
        for attr in ("mu_test", "mu_N", "running_mean"):
            if hasattr(self.shifter, attr):
                previous = getattr(self.shifter, attr)
                target_attr = attr
                break
        else:  # pragma: no cover
            return
        if previous is None:
            new_value = batch_mean  # mu_N(0) := mu_N(X_1), addendum
        else:
            new_value = alpha * batch_mean + (1.0 - alpha) * previous.to(batch_mean.device, batch_mean.dtype)
        try:
            setattr(self.shifter, target_attr, new_value.detach().clone())
        except Exception:  # pragma: no cover
            pass

    # -- forward passes ------------------------------------------------------
    def _forward(self, images: torch.Tensor, prompt_vector: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """One ``torch.no_grad`` forward of the frozen backbone with a prompt."""
        if prompt_vector is not None:
            self._set_prompt(prompt_vector)
        prompt_tensor = self._prompt_tensor()
        images = images.to(device=self.device, dtype=torch.float32) if self.device is not None else images
        with torch.no_grad():
            forward_with_features = getattr(self.model, "forward_with_features", None)
            if callable(forward_with_features):
                try:
                    out = forward_with_features(images, prompt=prompt_tensor)
                except TypeError:  # pragma: no cover - positional signature
                    out = forward_with_features(images, prompt_tensor)
            else:  # pragma: no cover - plain timm model
                out = self.model(images)
            parsed = _extract_output(out)
            if parsed["final_cls"] is None:
                raise RuntimeError("backbone did not return final-layer CLS features")
            if parsed["logits"] is None:
                parsed["logits"] = self._head(parsed["final_cls"])
        return parsed

    def _classify(self, final_cls: torch.Tensor) -> torch.Tensor:
        """Shift with the live state, then run the head."""
        with torch.no_grad():
            shifted = self._shift(final_cls)
            return self._head(shifted)

    # -- interval bookkeeping ------------------------------------------------
    def _start_window(self) -> None:
        """Ask the CMA optimizer for the K candidates of a new interval."""
        if self.optimizer is None:
            raise RuntimeError("FOA-I requires a CMA optimizer (optimizer.ask/tell)")
        candidates = self.optimizer.ask(self.population_size)
        candidates = np.asarray(candidates, dtype=np.float64).reshape(len(candidates), -1)
        self._pending_candidates = candidates
        self._window_shift_state = self._snapshot()
        self._window_index += 1
        if self.verbose:
            logger.info(
                "FOA-I window %d: %d candidates (dim=%d), shift state %s",
                self._window_index,
                candidates.shape[0],
                candidates.shape[1],
                "snapshotted" if self._window_shift_state is not None else "disabled",
            )

    def _candidate_window(
        self,
        index: int,
        candidate: np.ndarray,
        num_samples: int,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Per-layer window features + final-layer CLS for one candidate.

        V1: read the cached features (no forward pass).
        V2: re-forward the cached images for this candidate.
        """
        if self.variant == VARIANT_FEATURES:
            per_sample = []
            for item in self._buffer:
                record = item.get("cand", {}).get(index)
                if record is None:
                    continue
                per_sample.append(record)
            if len(per_sample) != len(self._buffer):
                self.stats.skipped_candidates += 1
                if len(per_sample) == 0:
                    return [], torch.empty(0, self.prompt_dim if self.prompt_dim else 1, device=self.device)
            stacked = _stack_features([rec["cls_features"] for rec in per_sample])
            finals = []
            for rec in per_sample:
                final = rec["final_cls"]
                if final.dim() == 1:
                    final = final.unsqueeze(0)
                finals.append(final)
            final_cls = torch.cat(finals, dim=0) if finals else torch.empty(0, 0, device=self.device)
            return stacked, final_cls

        # --- V2: images are stored, features are recomputed per candidate ---
        images = [item["image"] for item in self._buffer if item.get("image") is not None]
        if not images:
            return [], torch.empty(0, 0, device=self.device)
        batch = torch.cat(images, dim=0)
        parsed = self._forward(batch, prompt_vector=candidate)
        return parsed["cls_features"], parsed["final_cls"]

    def _interval_update(self) -> Optional[Dict[str, Any]]:
        """One CMA generation over the buffered window (Eqn. 5 + Eqn. 6)."""
        if not self._buffer or self._pending_candidates is None:
            return None
        candidates = self._pending_candidates
        num_samples = len(self._buffer)

        live_state = self._snapshot()
        if self._window_shift_state is not None:
            self._restore(self._window_shift_state)  # candidates share mu_N(t-1)

        values: List[float] = []
        try:
            for index, candidate in enumerate(candidates):
                cls_features, final_cls = self._candidate_window(index, candidate, num_samples)
                if final_cls is None or final_cls.numel() == 0:
                    values.append(float("inf"))
                    continue
                with torch.no_grad():
                    scoring_features = final_cls
                    if self.shift_candidates:
                        scoring_features = self._shift(final_cls)
                    logits = self._head(scoring_features)
                    if self.fitness is not None:
                        try:
                            terms = self.fitness.evaluate(logits, cls_features)
                        except TypeError:  # pragma: no cover - alternate signature
                            terms = self.fitness(logits, cls_features)
                        total = getattr(terms, "total", None)
                        if total is None and isinstance(terms, dict):
                            total = terms.get("total")
                        if total is None:
                            total = terms
                        values.append(float(total))
                    else:  # pragma: no cover - entropy-only fallback
                        probs = torch.softmax(logits.float(), dim=-1)
                        values.append(float(-(probs * torch.log(probs.clamp_min(1e-12))).sum(-1).sum()))
        finally:
            self._restore(live_state)  # the live per-sample shifting state continues

        values_arr = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values_arr)
        if not finite.any():
            logger.warning("FOA-I: all candidate fitness values were non-finite; skipping CMA update")
            self._buffer.clear()
            self._pending_candidates = None
            self._window_shift_state = None
            return None

        self.optimizer.tell(values, candidates)

        best_index = int(np.argmin(np.where(finite, values_arr, np.inf)))
        self.incumbent = np.asarray(candidates[best_index], dtype=np.float64).reshape(-1).copy()
        self._set_prompt(self.incumbent)  # next interval predicts with the new best prompt
        self.stats.updates += 1
        self._last_values = values
        self._last_best = best_index

        summary = {
            "window": self._window_index,
            "num_samples": num_samples,
            "variant": self.variant,
            "interval": self.interval,
            "best_index": best_index,
            "best_fitness": float(values_arr[best_index]),
            "mean_fitness": float(np.mean(values_arr[finite])),
            "min_fitness": float(np.min(values_arr[finite])),
            "max_fitness": float(np.max(values_arr[finite])),
        }
        if self.verbose:
            logger.info(
                "FOA-I window %d update: best=%d fitness=%.4f (mean %.4f) over %d samples",
                self._window_index,
                best_index,
                summary["best_fitness"],
                summary["mean_fitness"],
                num_samples,
            )

        self._buffer.clear()
        self._pending_candidates = None
        self._window_shift_state = None
        return summary

    # -- public API ----------------------------------------------------------
    def reset(self) -> None:
        """Reset the interval state (prompt/CMA/EMA state of the wrapped FOA too)."""
        reset_fn = getattr(self.foa, "reset", None)
        if callable(reset_fn):
            try:
                reset_fn()
            except Exception:
                pass
        self.stats = IntervalStats()
        self._buffer.clear()
        self._pending_candidates = None
        self._window_shift_state = None
        self._last_values = None
        self._last_best = None
        self._window_index = 0
        self.incumbent = self._initial_incumbent()

    def adapt_sample(self, image: Any, target: Optional[Any] = None) -> Dict[str, Any]:
        """Process one test sample: predict, shift/EMA-update, buffer, maybe update."""
        images = _to_image_tensor(image, self.device)
        if images.shape[0] != 1:
            # the interval protocol is defined per sample; loop over a batch
            results = [self.adapt_sample(images[i : i + 1], None) for i in range(images.shape[0])]
            logits = torch.cat([r["logits"] for r in results], dim=0)
            out = {
                "logits": logits,
                "pred": logits.argmax(dim=-1),
                "target": target,
                "interval_update": results[-1]["interval_update"],
            }
            return out

        if self._pending_candidates is None:
            self._start_window()

        # 1) predict with the incumbent prompt (live activation-shifting state)
        self._set_prompt(self.incumbent)
        incumbent_out = self._forward(images)
        final_cls = incumbent_out["final_cls"]
        logits = self._classify(final_cls)
        # Eqn. (9) uses the un-shifted final-layer CLS feature of x_t
        self._update_shift(final_cls)

        # 2) buffer the window data
        if self.variant == VARIANT_FEATURES:
            candidate_records: Dict[int, Dict[str, Any]] = {}
            for index, candidate in enumerate(self._pending_candidates):
                candidate_records[index] = self._forward(images, prompt_vector=candidate)
            self._buffer.append({"target": target, "cand": candidate_records})
        else:
            self._buffer.append({"target": target, "image": images.detach()})

        self.stats.samples_seen += 1
        self._set_prompt(self.incumbent)

        # 3) interval boundary -> one CMA update over the window
        update_summary = None
        if len(self._buffer) >= self.interval:
            update_summary = self._interval_update()

        return {
            "logits": logits,
            "pred": logits.argmax(dim=-1),
            "target": target,
            "interval_update": update_summary,
        }

    # alias mirroring the FOA naming
    __call__ = adapt_sample

    def flush(self) -> Optional[Dict[str, Any]]:
        """Run a final CMA update on a leftover partial window (stream end)."""
        if not self.update_partial_buffer or not self._buffer:
            return None
        if self._pending_candidates is None:
            self._start_window()
        return self._interval_update()

    def run(
        self,
        stream: Iterable[Any],
        accumulator: Optional[Any] = None,
        verbose: bool = False,
        log_every: int = 100,
        max_samples: Optional[int] = None,
        flush: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run FOA-I over an ordered single-sample test stream."""
        acc = accumulator if accumulator is not None else _build_accumulator(getattr(self.foa, "cfg", None))
        verbose = bool(verbose or self.verbose)
        do_flush = self.update_partial_buffer if flush is None else bool(flush)
        start = time.time()
        updates: List[Dict[str, Any]] = []
        num_samples = 0

        for batch in stream:
            images, targets = unpack_batch(batch)
            images = _to_image_tensor(images, self.device)
            if targets is not None:
                targets = torch.as_tensor(
                    targets.detach().cpu().numpy().reshape(-1) if isinstance(targets, torch.Tensor) else np.asarray(targets).reshape(-1)
                )
            for i in range(images.shape[0]):
                sample_target = None if targets is None else int(targets[i].item())
                result = self.adapt_sample(images[i : i + 1], sample_target)
                num_samples += 1
                if sample_target is not None:
                    try:
                        acc.update(result["logits"], torch.as_tensor([sample_target]))
                    except Exception:  # pragma: no cover - metric signature variations
                        pass
                if result.get("interval_update"):
                    updates.append(result["interval_update"])
                if verbose and log_every and num_samples % log_every == 0:
                    logger.info(
                        "FOA-I [%s I=%d] samples=%d updates=%d elapsed=%.1fs",
                        self.variant,
                        self.interval,
                        num_samples,
                        self.stats.updates,
                        time.time() - start,
                    )
                if max_samples is not None and num_samples >= max_samples:
                    break
            if max_samples is not None and num_samples >= max_samples:
                break

        if do_flush:
            tail = self.flush()
            if tail:
                updates.append(tail)

        summary = acc.compute() if hasattr(acc, "compute") else {"accuracy": float("nan"), "ece": float("nan"), "num_samples": num_samples}
        out: Dict[str, Any] = {
            "method": f"foa-i-{self.variant}",
            "variant": self.variant,
            "interval": self.interval,
            "accuracy": float(summary.get("accuracy", float("nan"))),
            "ece": float(summary.get("ece", float("nan"))),
            "num_samples": int(summary.get("num_samples", num_samples) or num_samples),
            "num_updates": int(self.stats.updates),
            "population_size": int(self.population_size),
            "prompt_dim": int(self.prompt_dim),
            "shifting_enabled": self.shifter is not None,
            "wall_clock_s": time.time() - start,
            "window_updates": updates,
            "stats": self.stats.to_dict(),
        }
        if verbose:
            logger.info(
                "FOA-I [%s I=%d] done: acc=%.2f%% ece=%.2f%% (%d samples, %d updates, %.1fs)",
                self.variant,
                self.interval,
                out["accuracy"],
                out["ece"],
                out["num_samples"],
                out["num_updates"],
                out["wall_clock_s"],
            )
        return out

    # -- persistence ---------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "variant": self.variant,
            "interval": self.interval,
            "incumbent": np.asarray(self.incumbent).copy(),
            "stats": self.stats.to_dict(),
            "window_index": self._window_index,
            "population_size": self.population_size,
        }
        if self.optimizer is not None and hasattr(self.optimizer, "state_dict"):
            try:
                state["optimizer"] = self.optimizer.state_dict()
            except Exception:
                pass
        shift_state = self._snapshot()
        if isinstance(shift_state, dict) and "attr" not in shift_state:
            state["shifter"] = shift_state
        return state

    def load_state_dict(self, state: Dict[str, Any], strict: bool = False) -> None:
        if not state:
            return
        if state.get("variant"):
            self.variant = normalize_variant(state["variant"])
        if state.get("interval"):
            self.interval = int(state["interval"])
        if state.get("incumbent") is not None:
            self.incumbent = np.asarray(state["incumbent"], dtype=np.float64).reshape(-1).copy()
            self._set_prompt(self.incumbent)
        if state.get("stats"):
            self.stats = IntervalStats(**{k: int(v) for k, v in state["stats"].items() if k in IntervalStats().to_dict()})
        self._window_index = int(state.get("window_index", 0))
        if state.get("optimizer") is not None and self.optimizer is not None and hasattr(self.optimizer, "load_state_dict"):
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:
                pass
        if state.get("shifter") is not None:
            self._restore(state["shifter"])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"FOAInterval(variant={self.variant!r}, interval={self.interval}, "
            f"population_size={self.population_size}, prompt_dim={self.prompt_dim}, "
            f"samples_seen={self.stats.samples_seen}, updates={self.stats.updates})"
        )


FOAI = FOAInterval  # alias


# ---------------------------------------------------------------------------
# Factories / drivers
# ---------------------------------------------------------------------------
def build_foa_i(
    cfg: Any = None,
    source_stats: Any = None,
    model: Any = None,
    device: Any = None,
    *,
    interval: Optional[int] = None,
    variant: Optional[str] = None,
    foa: Any = None,
    update_partial_buffer: bool = True,
    shift_candidates: bool = True,
    verbose: bool = False,
    **foa_kwargs: Any,
) -> FOAInterval:
    """Build a FOA-I runner, reusing the standard FOA construction path."""
    if foa is None:
        if build_foa is None:  # pragma: no cover
            raise ImportError("src.method.foa is unavailable; cannot build a FOA runner")
        foa = build_foa(cfg=cfg, source_stats=source_stats, model=model, device=device, **foa_kwargs)
    return FOAInterval(
        foa,
        interval=_resolve_interval(cfg, interval),
        variant=_resolve_variant(cfg, variant),
        update_partial_buffer=update_partial_buffer,
        shift_candidates=shift_candidates,
        device=device,
        verbose=verbose,
    )


def _build_single_sample_loader(cfg: Any, loader: Any = None, corruption: Optional[str] = None, limit_batches: Optional[int] = None) -> Any:
    """Build/limit an ordered BS=1 stream (ImageNet-C style) for FOA-I."""
    if loader is None:
        try:
            try:
                from ...scripts.run_foa import build_test_loader  # type: ignore
            except Exception:
                from scripts.run_foa import build_test_loader  # type: ignore

            loader = build_test_loader(cfg, corruption=corruption, batch_size=1)
        except Exception as exc:  # pragma: no cover - fall back to the data package
            try:
                from ..data.datasets import build_dataset_loader  # type: ignore

                loader = build_dataset_loader(cfg, corruption=corruption, batch_size=1)
            except Exception:
                raise RuntimeError(f"could not build a BS=1 test loader: {exc}") from exc
    if limit_batches is not None:
        loader = _MaybeLimited(loader, limit_batches)
    return loader


class _MaybeLimited:
    """Cap the number of batches consumed from a loader."""

    def __init__(self, loader: Any, limit: Optional[int] = None) -> None:
        self.loader = loader
        self.limit = limit
        self.dataset = getattr(loader, "dataset", None)
        self.batch_size = getattr(loader, "batch_size", 1)

    def __iter__(self) -> Iterator[Any]:
        count = 0
        for batch in self.loader:
            if self.limit is not None and count >= self.limit:
                break
            yield batch
            count += 1

    def __len__(self) -> int:
        try:
            total = len(self.loader)
        except Exception:  # pragma: no cover
            return self.limit or 0
        return min(total, self.limit) if self.limit is not None else total


def run_foa_i(
    cfg: Any = None,
    loader: Any = None,
    *,
    interval: Optional[int] = None,
    variant: Optional[str] = None,
    corruption: Optional[str] = None,
    device: Any = None,
    limit_batches: Optional[int] = None,
    max_samples: Optional[int] = None,
    source_stats: Any = None,
    foa: Any = None,
    verbose: bool = True,
    **foa_kwargs: Any,
) -> Dict[str, Any]:
    """Run FOA-I (one interval/variant) over a single-sample stream.

    Returns a Table-6 style summary dict: ``accuracy``, ``ece``, ``interval``,
    ``variant``, ``num_samples`` and bookkeeping.
    """
    if loader is None:
        loader = _build_single_sample_loader(cfg, loader=None, corruption=corruption, limit_batches=limit_batches)
    runner = build_foa_i(
        cfg=cfg,
        source_stats=source_stats,
        model=None,
        device=device,
        interval=interval,
        variant=variant,
        foa=foa,
        verbose=verbose,
        **foa_kwargs,
    )
    result = runner.run(loader, verbose=verbose, max_samples=max_samples)
    result["corruption"] = corruption
    result["dataset"] = _cfg_get(cfg, "data.dataset", default=None)
    return result


def run_interval_sweep(
    cfg: Any = None,
    intervals: Sequence[int] = FOA_I_INTERVALS,
    variants: Sequence[str] = (VARIANT_FEATURES,),
    *,
    corruption: Optional[str] = None,
    device: Any = None,
    limit_batches: Optional[int] = None,
    max_samples: Optional[int] = None,
    loader: Any = None,
    verbose: bool = True,
    **foa_kwargs: Any,
) -> Dict[str, Any]:
    """Sweep the intervals (and variants) of Table 6, returning per-setting results."""
    results: Dict[str, Any] = {"per_setting": {}, "intervals": list(intervals), "variants": [normalize_variant(v) for v in variants]}
    for variant_name in variants:
        canonical = normalize_variant(variant_name)
        for interval in intervals:
            loader_i = loader if loader is not None else _build_single_sample_loader(
                cfg, loader=None, corruption=corruption, limit_batches=limit_batches
            )
            outcome = run_foa_i(
                cfg,
                loader=loader_i,
                interval=int(interval),
                variant=canonical,
                corruption=corruption,
                device=device,
                limit_batches=limit_batches,
                max_samples=max_samples,
                verbose=verbose,
                **foa_kwargs,
            )
            results["per_setting"][f"{canonical}_I{int(interval)}"] = outcome
            if verbose:
                print(
                    f"FOA-I [{canonical}] I={int(interval):>2d}: "
                    f"acc={outcome['accuracy']:.2f}% ece={outcome['ece']:.2f}% "
                    f"(updates={outcome['num_updates']}, {outcome['wall_clock_s']:.1f}s)"
                )
    return results


def print_table6(results: Dict[str, Any]) -> None:
    """Pretty-print FOA-I results next to the paper's Table 6 reference values."""
    per_setting = results.get("per_setting", {}) if "per_setting" in results else results
    print("\nTable 6 -- FOA-I for single-sample adaptation (ImageNet-C, Gaussian, level 5)")
    print("setting        Acc.(%)  ECE(%)   paper Acc./ECE")
    noadapt = TABLE6_REFERENCE["noadapt"]
    tent = TABLE6_REFERENCE["tent_bs64"]
    print(f"NoAdapt        {noadapt['accuracy']:7.1f}  {noadapt['ece']:6.1f}   (reference)")
    print(f"TENT (BS=64)   {tent['accuracy']:7.1f}  {tent['ece']:6.1f}   (reference)")
    for key, value in per_setting.items():
        interval = int(value.get("interval", 0))
        ref = TABLE6_REFERENCE["foa_i"].get(interval)
        ref_text = f"{ref['accuracy']:.1f}/{ref['ece']:.1f}" if ref else "-"
        print(f"{key:<14s} {value['accuracy']:7.2f}  {value['ece']:6.2f}   {ref_text}")
    print()


# ---------------------------------------------------------------------------
# CLI (small convenience wrapper; the sweep scripts also import this module)
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FOA-I: interval-update FOA for BS=1 streams")
    parser.add_argument("--config", required=True, help="YAML config (e.g. configs/foa_imagenetc.yaml)")
    parser.add_argument("--extra-config", default=None, help="optional second YAML merged on top")
    parser.add_argument("--interval", type=int, default=None, help="interval I (default 4)")
    parser.add_argument("--variant", default=None, choices=list(FOA_I_VARIANTS), help="v1 = cache features, v2 = cache images")
    parser.add_argument("--sweep", action="store_true", help="sweep all of I in {4,8,16,32,64}")
    parser.add_argument("--corruption", default=None, help="ImageNet-C corruption (default: config value)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--source-stats", default=None, help="override source statistics checkpoint")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        try:
            from ..utils.config import load_config, save_config, config_to_dict  # type: ignore
        except Exception:
            from src.utils.config import load_config, save_config, config_to_dict  # type: ignore

        cfg = load_config(args.config) if args.extra_config is None else load_config(args.config, args.extra_config)
        if args.source_stats:
            try:
                cfg.source_stats.path = args.source_stats
            except Exception:  # pragma: no cover
                cfg.setdefault("source_stats", {})["path"] = args.source_stats
    except Exception as exc:  # pragma: no cover - config optional for smoke tests
        logger.warning("could not load config %s (%s); continuing with defaults", args.config, exc)
        cfg = None

    if args.sweep:
        results = run_interval_sweep(
            cfg,
            intervals=FOA_I_INTERVALS,
            variants=(args.variant or VARIANT_FEATURES,),
            corruption=args.corruption,
            device=args.device,
            limit_batches=args.limit_batches,
            max_samples=args.max_samples,
            verbose=not args.quiet,
        )
        print_table6(results)
    else:
        results = run_foa_i(
            cfg,
            interval=args.interval,
            variant=args.variant,
            corruption=args.corruption,
            device=args.device,
            limit_batches=args.limit_batches,
            max_samples=args.max_samples,
            verbose=not args.quiet,
        )
        print(
            f"FOA-I [{results['variant']}] I={results['interval']}: "
            f"acc={results['accuracy']:.2f}% ece={results['ece']:.2f}%"
        )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        try:
            payload = {"config": config_to_dict(cfg) if cfg is not None else None, "results": results}
        except Exception:  # pragma: no cover
            payload = {"config": None, "results": results}
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
