"""Centered Kernel Alignment (CKA) for representation-drift analysis.

Reproduces the analysis of Section §B.3 / Figure 26 / Figure 27 of

    Wołczyk et al. (2024), "Fine-tuning Reinforcement Learning Models is
    Secretly a Forgetting Mitigation Problem".

The paper uses Central Kernel Alignment (Kornblith et al., 2019) to study
similarity of representations:

    CKA(K, L) = HSIC(K, L) / sqrt( HSIC(K, K) HSIC(L, L) ),        (Eq. B.3.1)

where ``K_ij = k(x_i, x_j)`` and ``L_ij = l(y_i, y_j)`` are Gram matrices of
the activations ``X in R^{n x p_1}`` and ``Y in R^{n x p_2}`` recorded on the
*same n examples*, and ``k``/``l`` are kernels -- in the paper "we simply use a
linear kernel in both cases".

This module provides

* the exact linear-kernel CKA of Eq. (B.3.1) with the (unbiased) HSIC
  estimator of Gretton et al. (2005) -- default is the normalized, biased
  ``HSIC_1`` form used in practise, which for linear kernels reduces to a
  closed-form expression in terms of the centered Gram matrices,
* per-layer activation extraction from an arbitrary policy network (the
  RoboticSequence SAC actor built in ``src/robotic_sequence/sac.py`` and the
  nets from ``src/robotic_sequence/heads.py``),
* :class:`CKATracker`, an online accumulator that compares the *current*
  network's activations against the pre-trained (``pi_*``) activations on the
  same fixed batch of states -- reproducing Figure 27 ("CKA shows later policy
  layers change more than early ones, with partial recovery as FAR tasks are
  revisited"),
* helpers to quantify layer-wise drift / recovery and to plot the
  layer-vs-step heat map.

Everything is dependency-light: only NumPy (required) and PyTorch (optional --
imported lazily, only the activation-extraction helpers need it).  Matplotlib
is optional and only used inside the plotting helper.

CLI::

    python -m src.analysis.cka --checkpoint ckpt.pt --data layer_inputs.npz \
        --output-dir results/cka
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

try:  # pragma: no cover - optional at import time
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_KERNEL = "linear"
DEFAULT_CONFIDENCE = 0.90

#: Names of the functions this module exposes through ``__all__``.
__all__ = [
    "linear_kernel",
    "rbf_kernel",
    "gram_matrix",
    "hsic",
    "cka",
    "cka_from_kernels",
    "batch_cka",
    "layer_activations",
    "activation_snapshots",
    "layer_names",
    "resolve_layer_name",
    "LayerDrift",
    "layer_drift",
    "recovery_score",
    "CKATracker",
    "compute_layerwise_cka",
    "plot_cka_heatmap",
    "main",
    "DEFAULT_KERNEL",
]


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _require_numpy() -> Any:
    if np is None:  # pragma: no cover
        raise RuntimeError(
            "NumPy is required for CKA computations but is not installed."
        )
    return np


def _as_2d(x: Any) -> Any:
    """Coerce an array-like to a float64 2-D matrix ``(n_samples, n_features)``."""

    np_ = _require_numpy()
    arr = np_.asarray(x, dtype=np_.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def linear_kernel(x: Any, y: Optional[Any] = None) -> Any:
    """Linear kernel Gram matrix ``K = X Y^T`` (Eq. B.3.1, ``k(x, y) = <x, y>``).

    Parameters
    ----------
    x:
        Array ``(n, p_1)``.
    y:
        Optional array ``(m, p_2)``.  Defaults to ``x`` (``m = n``).
    """

    np_ = _require_numpy()
    x = _as_2d(x)
    if y is None:
        y = x
    else:
        y = _as_2d(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            "linear_kernel requires matching feature dimensions, got "
            f"{x.shape[1]} and {y.shape[1]}"
        )
    return x @ y.T


def rbf_kernel(x: Any, y: Optional[Any] = None, sigma: Optional[float] = None) -> Any:
    """Gaussian (RBF) kernel, provided as an alternative to the paper's linear one."""

    np_ = _require_numpy()
    x = _as_2d(x)
    y = x if y is None else _as_2d(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError("rbf_kernel requires matching feature dimensions")
    x_norm = (x ** 2).sum(axis=1, keepdims=True)
    y_norm = (y ** 2).sum(axis=1, keepdims=True).T
    sq = x_norm + y_norm - 2.0 * (x @ y.T)
    sq = np_.maximum(sq, 0.0)
    if sigma is None:
        # median heuristic on the pairwise distances
        finite = sq[sq > 0]
        sigma = float(np_.sqrt(np_.median(finite))) if finite.size else 1.0
        sigma = sigma if sigma > 0 else 1.0
    return np_.exp(-sq / (2.0 * sigma ** 2))


def gram_matrix(
    x: Any,
    y: Optional[Any] = None,
    kernel: Union[str, Callable[..., Any]] = DEFAULT_KERNEL,
    **kernel_kwargs: Any,
) -> Any:
    """Build a Gram matrix for ``kernel in {"linear", "rbf", callable}``."""

    if callable(kernel):
        return _as_2d(kernel(x, y) if y is not None else kernel(x))
    name = str(kernel).strip().lower()
    if name in ("linear", "lin", "dot"):
        return linear_kernel(x, y)
    if name in ("rbf", "gaussian", "gauss", "sqexp"):
        return rbf_kernel(x, y, **kernel_kwargs)
    raise ValueError(f"Unknown kernel '{kernel}' (expected 'linear', 'rbf' or a callable)")


def _center(gram: Any) -> Any:
    """Column/row-center a Gram matrix (the ``H K H`` of the HSIC estimator)."""

    np_ = _require_numpy()
    n = gram.shape[0]
    if n == 0:
        return gram
    row_mean = gram.mean(axis=1, keepdims=True)
    col_mean = gram.mean(axis=0, keepdims=True)
    total_mean = gram.mean()
    return gram - row_mean - col_mean + total_mean


# ---------------------------------------------------------------------------
# HSIC / CKA
# ---------------------------------------------------------------------------


def hsic(
    k: Any,
    l: Optional[Any] = None,
    kernel: Union[str, Callable[..., Any]] = DEFAULT_KERNEL,
    estimator: str = "biased",
    **kernel_kwargs: Any,
) -> float:
    """Hilbert-Schmidt Independence Criterion of Gretton et al. (2005).

    ``estimator="biased"`` (default) is the ``HSIC_1`` form
    ``tr(K H L H) / n^2`` used by Kornblith et al. (2019) (and hence by the
    paper's Eq. B.3.1).  ``estimator="unbiased"`` uses the ``HSIC_2`` form with
    zero diagonal.

    ``k``/``l`` may be either raw activation matrices or precomputed Gram
    matrices -- pass ``kernel="precomputed"`` for the latter.
    """

    np_ = _require_numpy()
    name = str(kernel).strip().lower()
    if name in ("precomputed", "gram", "kernel"):
        K = _as_2d(k)
        L = K if l is None else _as_2d(l)
    else:
        K = gram_matrix(k, None, kernel, **kernel_kwargs)
        L = K if l is None else gram_matrix(l, None, kernel, **kernel_kwargs)

    if K.shape[0] != L.shape[0]:
        raise ValueError(
            "HSIC requires the same number of examples, got "
            f"{K.shape[0]} and {L.shape[0]}"
        )
    n = K.shape[0]
    if n < 2:
        return 0.0
    if estimator in ("unbiased", "hsic2", "hsic_2"):
        K = K.copy()
        L = L.copy()
        np_.fill_diagonal(K, 0.0)
        np_.fill_diagonal(L, 0.0)
        term = float(np_.trace(K @ L))
        return term / (n * (n - 3))
    Kc = _center(K)
    Lc = _center(L)
    return float(np_.sum(Kc * Lc) / (n * n))


def cka_from_kernels(
    K: Any,
    L: Any,
    estimator: str = "biased",
) -> float:
    """``CKA(K, L)`` from two precomputed Gram matrices (Eq. B.3.1)."""

    np_ = _require_numpy()
    K = _as_2d(K)
    L = _as_2d(L)
    if K.shape[0] != L.shape[0]:
        raise ValueError("CKA requires equally many examples in both Gram matrices")
    denom = np_.sqrt(max(hsic(K, None, kernel="precomputed", estimator=estimator), 0.0)
                     * max(hsic(L, None, kernel="precomputed", estimator=estimator), 0.0))
    if not np_.isfinite(denom) or denom <= 0.0:
        # Degenerate (e.g. constant features): similarity is undefined -> 0.
        return 0.0
    return float(hsic(K, L, kernel="precomputed", estimator=estimator) / denom)


def cka(
    x: Any,
    y: Any,
    kernel: Union[str, Callable[..., Any]] = DEFAULT_KERNEL,
    estimator: str = "biased",
    return_gram: bool = False,
    **kernel_kwargs: Any,
) -> Union[float, Tuple[float, Any, Any]]:
    """Linear-kernel CKA between activation matrices ``x`` and ``y``.

    ``x`` and ``y`` record activations for the *same* ``n`` examples, with
    possibly different numbers of neurons ``p_1``/``p_2`` -- exactly the setup
    of §B.3 (Eq. 1).  Kernels are centred before the HSIC, so the result is
    invariant to translation and (for a single layer) to rotation of the
    features, as expected from CKA.
    """

    K = gram_matrix(x, None, kernel, **kernel_kwargs)
    L = gram_matrix(y, None, kernel, **kernel_kwargs)
    value = cka_from_kernels(K, L, estimator=estimator)
    if return_gram:
        return value, K, L
    return value


def batch_cka(
    x: Any,
    y: Any,
    max_examples: Optional[int] = None,
    seed: int = 0,
    kernel: Union[str, Callable[..., Any]] = DEFAULT_KERNEL,
    **kwargs: Any,
) -> float:
    """CKA on at most ``max_examples`` rows (subsampled) -- memory friendly."""

    np_ = _require_numpy()
    x = _as_2d(x)
    y = _as_2d(y)
    n = min(x.shape[0], y.shape[0])
    if max_examples is not None and n > max_examples:
        rng = np_.random.default_rng(seed)
        idx = rng.choice(n, size=int(max_examples), replace=False)
        x = x[idx]
        y = y[idx]
    elif x.shape[0] != y.shape[0]:
        x = x[:n]
        y = y[:n]
    return float(cka(x, y, kernel=kernel, **kwargs))


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------


def layer_names(model: Any) -> List[str]:
    """Return the ordered names of the (non-parameter) submodules of ``model``.

    Linear/conv/normalisation blocks are returned in registration order, which
    is the order used by :func:`layer_activations` and by the paper's
    layer-by-layer analysis.
    """

    try:  # pragma: no cover - needs torch
        import torch.nn as nn
    except Exception:  # pragma: no cover
        return []
    names: List[str] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(
            module,
            (
                nn.Linear,
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.LayerNorm,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.GroupNorm,
                nn.ReLU,
                nn.LeakyReLU,
                nn.Tanh,
                nn.ELU,
                nn.Flatten,
            ),
        ):
            names.append(name)
    return names


def resolve_layer_name(model: Any, layer: Union[str, int, Any]) -> Any:
    """Resolve ``layer`` (name, index into :func:`layer_names`, or a module)."""

    if not isinstance(layer, str):
        if isinstance(layer, int):
            names = layer_names(model)
            if not names:
                raise ValueError("Model exposes no analysable submodules")
            return model.get_submodule(names[layer % len(names)])
        return layer
    try:  # pragma: no cover
        return model.get_submodule(layer)
    except AttributeError:
        pass
    # Fall back: substring match.
    matches = [n for n in layer_names(model) if layer in n]
    if not matches:
        raise KeyError(f"Layer '{layer}' not found in model")
    return model.get_submodule(matches[0])


def activation_snapshots(
    model: Any,
    obs: Any,
    layers: Optional[Sequence[Union[str, int, Any]]] = None,
    batch_size: Optional[int] = None,
    forward_fn: Optional[Callable[[Any, Any], Any]] = None,
    flatten: bool = True,
    detach: bool = True,
    device: Optional[Any] = None,
    **forward_kwargs: Any,
) -> Dict[str, Any]:
    """Record the activations of ``layers`` for the inputs ``obs``.

    Parameters
    ----------
    model:
        The network (typically ``SACPolicy`` -- forward must accept a batch of
        observations, optionally with a stage ID passed through
        ``forward_kwargs``).
    obs:
        Tensor/array of ``n`` observation vectors.
    layers:
        Names / indices / modules; defaults to every analysable submodule
        (see :func:`layer_names`).
    batch_size:
        Optional mini-batching to bound peak memory.
    forward_fn:
        Optional ``forward_fn(model, batch) -> output`` override for exotic
        model signatures.
    flatten:
        Flatten non-batch dimensions of the activations (required for a linear
        kernel over per-neuron activations).
    detach:
        Detach from the autograd graph (always true for analysis; default).

    Returns
    -------
    dict mapping layer name -> NumPy array ``(n, p)``.
    """

    np_ = _require_numpy()
    try:  # pragma: no cover - needs torch
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for activation extraction") from exc

    if layers is None:
        chosen: List[Any] = list(layer_names(model))
    else:
        chosen = list(layers)
    resolved = []
    for layer in chosen:
        module = resolve_layer_name(model, layer)
        name = layer if isinstance(layer, str) else (getattr(module, "_cka_name", None) or str(layer))
        resolved.append((str(name), module))

    captured: Dict[str, List[Any]] = {name: [] for name, _ in resolved}
    handles = []

    def _make_hook(key: str) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            out = output
            if isinstance(out, (tuple, list)):
                out = out[0]
            if flatten and hasattr(out, "flatten"):
                out = out.flatten(start_dim=1)
            if detach and hasattr(out, "detach"):
                out = out.detach()
            captured[key].append(out)

        return hook

    for name, module in resolved:
        handles.append(module.register_forward_hook(_make_hook(name)))

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    obs_t = obs
    if device is not None and hasattr(obs_t, "to"):
        obs_t = obs_t.to(device)
    if batch_size is None or not hasattr(obs_t, "__len__"):
        batches = [obs_t]
    else:
        total = len(obs_t)
        batches = [obs_t[i:i + int(batch_size)] for i in range(0, total, int(batch_size))]

    kwargs: Dict[str, Any] = {}
    for key, value in forward_kwargs.items():
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            try:
                chunks = [
                    value[i:i + int(batch_size or len(value))]
                    for i in range(0, len(value), int(batch_size or len(value)))
                ]
                kwargs[key] = chunks
                continue
            except Exception:  # pragma: no cover
                pass
        kwargs[key] = value

    with _no_grad():
        for batch_index, batch in enumerate(batches):
            call_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, list) and len(value) == len(batches):
                    call_kwargs[key] = value[batch_index]
                else:
                    call_kwargs[key] = value
            if forward_fn is not None:
                forward_fn(model, batch, **call_kwargs)
            else:
                model(batch, **call_kwargs)

    for handle in handles:
        handle.remove()
    if was_training and hasattr(model, "train"):
        model.train()

    out_dict: Dict[str, Any] = {}
    for name, chunks in captured.items():
        if not chunks:
            continue
        arrs = []
        for chunk in chunks:
            arr = chunk.detach().cpu().numpy() if hasattr(chunk, "detach") else np_.asarray(chunk)
            arrs.append(np_.asarray(arr, dtype=np_.float64))
        stacked = np_.concatenate(arrs, axis=0) if len(arrs) > 1 else arrs[0]
        out_dict[name] = stacked.reshape(stacked.shape[0], -1)
    return out_dict


def _no_grad() -> Any:
    """Context manager: ``torch.no_grad()`` if torch is available, else a no-op."""

    try:  # pragma: no cover
        import torch

        return torch.no_grad()
    except Exception:  # pragma: no cover
        import contextlib

        return contextlib.nullcontext()


def layer_activations(
    model: Any,
    obs: Any,
    layers: Optional[Sequence[Union[str, int, Any]]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias of :func:`activation_snapshots` (activations at each layer)."""

    return activation_snapshots(model, obs, layers=layers, **kwargs)


# ---------------------------------------------------------------------------
# Drift metrics
# ---------------------------------------------------------------------------


@dataclass
class LayerDrift:
    """CKA-based drift statistics for a single layer."""

    layer: str
    cka: float
    drift: float
    reference: Optional[float] = None
    step: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"layer": self.layer, "cka": self.cka, "drift": self.drift}
        if self.reference is not None:
            out["reference"] = self.reference
        if self.step is not None:
            out["step"] = self.step
        return out


def layer_drift(
    reference: Mapping[str, Any],
    current: Mapping[str, Any],
    kernel: Union[str, Callable[..., Any]] = DEFAULT_KERNEL,
    step: Optional[float] = None,
    max_examples: Optional[int] = None,
) -> Dict[str, LayerDrift]:
    """Per-layer ``CKA(activations_pre, activations_now)`` and ``1 - CKA`` drift.

    ``drift`` is ``1 - CKA``: 0 means the layer's representations are unchanged
    relative to the pre-trained network ``pi_*``, 1 means fully altered.  The
    paper reports (through Figure 27) that *later* policy layers drift more than
    *earlier* ones.
    """

    out: Dict[str, LayerDrift] = {}
    for name, ref in reference.items():
        if name not in current:
            continue
        ref_arr = _as_2d(ref)
        cur_arr = _as_2d(current[name])
        n = min(ref_arr.shape[0], cur_arr.shape[0])
        if n == 0:
            continue
        value = batch_cka(
            ref_arr[:n],
            cur_arr[:n],
            max_examples=max_examples,
            kernel=kernel,
        )
        out[name] = LayerDrift(
            layer=name, cka=float(value), drift=float(1.0 - value), step=step
        )
    return out


def recovery_score(
    worst: float,
    final: float,
    reference: float = 1.0,
    initial: Optional[float] = None,
) -> float:
    """How much of a layer's representation similarity was recovered.

    ``(final - worst) / (reference - worst)``, clipped to ``[0, 1]``; if
    ``initial`` is provided the denominator uses ``reference - min(worst, initial)``.
    Used to quantify "partial recovery as FAR tasks are revisited" (§B.3).
    """

    floor = worst if initial is None else min(worst, initial)
    denom = float(reference) - float(floor)
    if denom <= 0:
        return float("nan")
    value = (float(final) - float(floor)) / denom
    return float(min(max(value, 0.0), 1.0))


# ---------------------------------------------------------------------------
# Online tracker
# ---------------------------------------------------------------------------


class CKATracker:
    """Track layer-wise CKA of the fine-tuned network against ``pi_*``.

    The reference activations are computed once (at construction, from the
    *pre-trained* model on a fixed batch of states) and reused for every
    subsequent measurement, so the comparison is always on the *same* ``n``
    examples -- as required by Eq. (B.3.1).

    Typical use inside a fine-tuning loop::

        tracker = CKATracker(pretrained_actor, obs_batch,
                             layers=["net.0", "net.2", "last_linear"])
        ...
        tracker.record(step, agent.policy)      # every eval interval
        tracker.recovery()                      # per-layer recovery scores
    """

    def __init__(
        self,
        reference_model: Any,
        obs: Any,
        layers: Optional[Sequence[Union[str, int, Any]]] = None,
        kernel: Union[str, Callable[..., Any]] = DEFAULT_KERNEL,
        batch_size: Optional[int] = None,
        device: Optional[Any] = None,
        max_examples: Optional[int] = None,
        forward_kwargs: Optional[Mapping[str, Any]] = None,
        reference_activations: Optional[Mapping[str, Any]] = None,
        name: str = "cka",
    ) -> None:
        self.name = name
        self.kernel = kernel
        self.batch_size = batch_size
        self.device = device
        self.max_examples = max_examples
        self._forward_kwargs: Dict[str, Any] = dict(forward_kwargs or {})
        self._obs = obs
        self._layers = list(layers) if layers is not None else None

        if reference_activations is not None:
            self.reference: Dict[str, Any] = {
                k: _as_2d(v) for k, v in reference_activations.items()
            }
        else:
            self.reference = activation_snapshots(
                reference_model,
                obs,
                layers=self._layers,
                batch_size=batch_size,
                device=device,
                **self._forward_kwargs,
            )
        #: ``step -> {layer: (cka, drift)}``
        self.history: Dict[float, Dict[str, Tuple[float, float]]] = {}

    # -- bookkeeping ------------------------------------------------------
    @property
    def layers(self) -> List[str]:
        """Layer names being tracked."""

        return list(self.reference.keys())

    def __len__(self) -> int:
        return len(self.history)

    @property
    def last_step(self) -> Optional[float]:
        if not self.history:
            return None
        return max(self.history.keys())

    # -- measurement ------------------------------------------------------
    def record(
        self,
        step: Optional[float] = None,
        model: Any = None,
        activations: Optional[Mapping[str, Any]] = None,
        **forward_kwargs: Any,
    ) -> Dict[str, LayerDrift]:
        """Measure CKA(``pi_*`` layer, current layer) at ``step`` and store it."""

        if activations is None:
            if model is None:
                raise ValueError("Provide either `model` or `activations` to record()")
            kwargs = dict(self._forward_kwargs)
            kwargs.update(forward_kwargs)
            activations = activation_snapshots(
                model,
                self._obs,
                layers=self._layers,
                batch_size=self.batch_size,
                device=self.device,
                **kwargs,
            )
        drifts = layer_drift(
            self.reference,
            activations,
            kernel=self.kernel,
            step=step,
            max_examples=self.max_examples,
        )
        key = float(step) if step is not None else float(len(self.history))
        self.history[key] = {name: (d.cka, d.drift) for name, d in drifts.items()}
        return drifts

    def record_step(self, step: float, model: Any, **forward_kwargs: Any) -> Dict[str, LayerDrift]:
        """Convenience wrapper for ``record(step, model=model)``."""

        return self.record(step=step, model=model, **forward_kwargs)

    # -- summaries --------------------------------------------------------
    def curve(self, layer: str) -> Tuple[List[float], List[float]]:
        """``(steps, CKA)`` trajectory for one layer."""

        steps = sorted(self.history.keys())
        return steps, [self.history[s].get(layer, (float("nan"), float("nan")))[0] for s in steps]

    def drift_curve(self, layer: str) -> Tuple[List[float], List[float]]:
        """``(steps, drift = 1 - CKA)`` trajectory for one layer."""

        steps = sorted(self.history.keys())
        return steps, [self.history[s].get(layer, (float("nan"), float("nan")))[1] for s in steps]

    def final(self) -> Dict[str, float]:
        """CKA at the last recorded step, per layer."""

        if not self.history:
            return {}
        last = self.history[max(self.history.keys())]
        return {name: value[0] for name, value in last.items()}

    def worst(self) -> Dict[str, float]:
        """Minimum CKA over the run, per layer (deepest forgetting)."""

        out: Dict[str, float] = {}
        for layer in self.layers:
            values = [self.history[s][layer][0] for s in self.history if layer in self.history[s]]
            if values:
                out[layer] = float(min(values))
        return out

    def recovery(self, reference: float = 1.0) -> Dict[str, float]:
        """Per-layer :func:`recovery_score` between the worst point and the end."""

        worst = self.worst()
        final = self.final()
        return {
            layer: recovery_score(worst[layer], final.get(layer, worst[layer]), reference=reference)
            for layer in worst
        }

    def layer_ordering(self) -> List[str]:
        """Layer names sorted by final drift (most-changed last)."""

        final = self.final()
        return sorted(final.keys(), key=lambda name: final[name])

    def later_layers_drift_more(self) -> bool:
        """Checks the qualitative claim of Figure 27.

        Compares the mean drift of the first half of the (registration-ordered)
        layers against the second half; returns ``True`` when later layers were
        altered more than earlier ones.
        """

        names = [n for n in self.layers if n in self.final()]
        if len(names) < 2:
            return True
        half = len(names) // 2
        early = [1.0 - self.final()[n] for n in names[:half]]
        late = [1.0 - self.final()[n] for n in names[half:]]
        return float(sum(late) / len(late)) >= float(sum(early) / len(early))

    # -- persistence ------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """Serialisable state (history + reference Gram matrices are omitted)."""

        return {
            "name": self.name,
            "kernel": self.kernel if isinstance(self.kernel, str) else str(self.kernel),
            "layers": self.layers,
            "history": {str(k): v for k, v in self.history.items()},
        }

    def save(self, path: str) -> str:
        """Write the tracked history to ``path`` as JSON."""

        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.state_dict(), handle, indent=2)
        return path

    def to_dict(self) -> Dict[str, Any]:
        """Full report: history, final/worst CKA, recovery scores."""

        return {
            "layers": self.layers,
            "history": {str(k): v for k, v in self.history.items()},
            "final": self.final(),
            "worst": self.worst(),
            "recovery": self.recovery(),
            "later_layers_drift_more": self.later_layers_drift_more(),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CKATracker(name={self.name!r}, layers={len(self.layers)}, "
            f"measurements={len(self.history)})"
        )


# ---------------------------------------------------------------------------
# Offline computation
# ---------------------------------------------------------------------------


def compute_layerwise_cka(
    reference: Any,
    candidates: Any,
    layers: Optional[Sequence[Union[str, int, Any]]] = None,
    kernel: Union[str, Callable[..., Any]] = DEFAULT_KERNEL,
    batch_size: Optional[int] = None,
    max_examples: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """CKA between a reference activation set and several candidate sets.

    ``candidates`` may be

    * a model -- its activations are computed on the same inputs as the
      reference (the reference ``obs`` must then be supplied through
      ``reference`` as ``(model, obs)``),
    * a mapping ``name -> activations``, or
    * a mapping ``name -> model`` paired with ``reference`` being a mapping of
      activations (activations are computed on ``reference_obs`` when given).

    Returns ``{candidate_name: {layer: cka}}``.
    """

    if isinstance(reference, tuple) and len(reference) == 2:
        ref_model, obs = reference
        ref_acts = activation_snapshots(
            ref_model, obs, layers=layers, batch_size=batch_size
        )
    else:
        ref_acts = {k: _as_2d(v) for k, v in dict(reference).items()}

    results: Dict[str, Dict[str, Any]] = {}
    if isinstance(candidates, Mapping):
        items = list(candidates.items())
    else:
        items = [("candidate", candidates)]
    for name, candidate in items:
        if isinstance(candidate, Mapping):
            acts = {k: _as_2d(v) for k, v in candidate.items()}
        else:
            acts = activation_snapshots(
                candidate, obs if isinstance(reference, tuple) else None,
                layers=layers, batch_size=batch_size,
            )
        drifts = layer_drift(
            ref_acts, acts, kernel=kernel, max_examples=max_examples
        )
        results[str(name)] = {layer: d.cka for layer, d in drifts.items()}
    return results


def aggregate_cka(values: Sequence[float], confidence: float = DEFAULT_CONFIDENCE) -> Dict[str, float]:
    """``{mean, half_width, n, std}`` with a normal-approximation CI (>=20 seeds)."""

    np_ = _require_numpy()
    arr = np_.asarray([v for v in values if v is not None and np_.isfinite(v)], dtype=np_.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "half_width": float("nan"), "n": 0, "std": float("nan")}
    mean = float(arr.mean())
    if arr.size == 1:
        return {"mean": mean, "half_width": 0.0, "n": 1, "std": 0.0}
    std = float(arr.std(ddof=1))
    half = _z_for(confidence) * std / float(np_.sqrt(arr.size))
    return {"mean": mean, "half_width": float(half), "n": int(arr.size), "std": std}


def _z_for(confidence: float) -> float:
    """Two-sided normal quantile for ``confidence`` (0.90 -> 1.6449)."""

    try:
        from math import erf, sqrt  # noqa: WPS433

        target = 0.5 * (1.0 + float(confidence))
        lo, hi = 0.0, 10.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    except Exception:  # pragma: no cover
        return 1.6449


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_cka_heatmap(
    tracker: Union[CKATracker, Mapping[str, Any]],
    path: Optional[str] = None,
    title: str = "CKA with pre-trained representations",
    cmap: str = "viridis",
    show: bool = False,
) -> Any:
    """Heat map of per-layer CKA over training steps (Figure 27 style).

    Rows are layers (registration order, i.e. early -> late), columns are
    evaluation steps, values are ``CKA(current, pi_*)``.
    """

    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np_ = _require_numpy()
    history = tracker.history if isinstance(tracker, CKATracker) else dict(tracker)
    if not history:
        raise ValueError("Nothing to plot: CKA history is empty")
    steps = sorted(history.keys())
    layers = list(tracker.layers) if isinstance(tracker, CKATracker) else sorted(
        {name for value in history.values() for name in value}
    )
    matrix = np_.full((len(layers), len(steps)), np_.nan)
    for j, step in enumerate(steps):
        for i, layer in enumerate(layers):
            entry = history[step].get(layer)
            if entry is None:
                continue
            matrix[i, j] = entry[0] if isinstance(entry, (tuple, list)) else entry

    fig, ax = plt.subplots(figsize=(max(4.0, 0.4 * len(steps) + 2), max(2.5, 0.4 * len(layers) + 1.5)))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([f"{int(s)}" if float(s).is_integer() else f"{s:g}" for s in steps], rotation=45, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("layer (early -> late)")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="CKA")
    fig.tight_layout()
    if path:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(path, dpi=150)
    if show:  # pragma: no cover
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Layer-wise CKA between a pre-trained and a fine-tuned network."
    )
    parser.add_argument("--reference", type=str, default=None,
                        help="checkpoint (state dict) of the pre-trained pi_* model")
    parser.add_argument("--checkpoint", type=str, nargs="+", default=None,
                        help="one or more fine-tuned checkpoints to compare")
    parser.add_argument("--data", type=str, default=None,
                        help=".npz/.npy file with the observation batch (`obs` key for npz)")
    parser.add_argument("--layers", type=str, nargs="*", default=None,
                        help="layer names/indices to analyse (default: all submodules)")
    parser.add_argument("--kernel", type=str, default=DEFAULT_KERNEL, choices=["linear", "rbf"])
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--plot", action="store_true", help="save a drift bar plot")
    return parser


def _load_obs(path: str) -> Any:
    np_ = _require_numpy()
    if path.endswith(".npz"):
        with np_.load(path) as data:
            key = "obs" if "obs" in data else sorted(data.files)[0]
            return data[key]
    return np_.load(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: compute CKA(s) against a pre-trained reference."""

    args = build_parser().parse_args(argv)
    if args.reference is None or args.checkpoint is None or args.data is None:
        build_parser().print_help()
        return 2

    try:  # pragma: no cover - needs torch
        import torch
    except Exception:  # pragma: no cover
        raise SystemExit("PyTorch is required to run the CKA CLI")

    from src.robotic_sequence.sac import build_sac_agent  # local import (torch needed)

    obs = _load_obs(args.data)
    # Build a reference agent lazily from the checkpoint's own shapes.
    ref_state = torch.load(args.reference, map_location="cpu")

    def _agent_from_state(state: Mapping[str, Any], obs_arr: Any) -> Any:
        net = state.get("policy", state)
        first = None
        for key, value in net.items():
            if key.endswith("weight") and value.dim() == 2:
                first = value
                break
        if first is None:  # pragma: no cover
            raise ValueError("Could not infer the policy input dimension from the checkpoint")
        obs_dim = int(first.shape[1])
        last = None
        for key, value in net.items():
            if key.endswith("weight") and value.dim() == 2:
                last = value
        action_dim = int(last.shape[0] // 2) if last is not None and last.shape[0] % 2 == 0 else 1
        # stage_id may have been appended to the observation (see config)
        if hasattr(obs_arr, "shape") and obs_arr.shape[-1] != obs_dim:
            obs_dim = int(obs_arr.shape[-1])
        agent = build_sac_agent({}, obs_dim=obs_dim, action_dim=max(action_dim, 1))
        agent.load_pretrained(state.get("actor", state.get("policy", state)))
        return agent

    ref_agent = _agent_from_state(ref_state, obs)
    try:
        ref_agent.policy.load_state_dict(ref_state.get("actor", ref_state.get("policy", ref_state)), strict=False)
    except Exception:  # pragma: no cover
        pass

    tracker = CKATracker(
        ref_agent.policy,
        obs,
        layers=args.layers,
        kernel=args.kernel,
        max_examples=args.max_examples,
    )

    report: Dict[str, Any] = {"kernel": args.kernel, "layers": tracker.layers, "checkpoints": {}}
    for index, path in enumerate(args.checkpoint):
        state = torch.load(path, map_location="cpu")
        cand = _agent_from_state(state, obs)
        try:
            cand.policy.load_state_dict(state.get("actor", state.get("policy", state)), strict=False)
        except Exception:  # pragma: no cover
            pass
        drifts = tracker.record(step=index, model=cand.policy)
        report["checkpoints"][path] = {name: d.cka for name, d in drifts.items()}

    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "cka.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        if args.plot:
            plot_cka_heatmap(tracker, path=os.path.join(output_dir, "cka_heatmap.png"))
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
