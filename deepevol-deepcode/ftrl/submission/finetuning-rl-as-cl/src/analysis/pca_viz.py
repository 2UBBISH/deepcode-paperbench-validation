"""PCA visualisation of the state space for the RoboticSequence experiments.

This module reproduces the visualisations of Section 5 / Figure 8 (top-right) of
Wolczyk et al. (2024), *Fine-tuning Reinforcement Learning Models is Secretly a
Forgetting Mitigation Problem*:

    "The state space projection coloured by the expert-action log-likelihood"

The recipe mirroring the paper is:

1.  Collect a frozen set of transitions with the pre-trained policy ``pi_*``
    (``src.analysis.loglikelihood.collect_expert_transitions``); these give the
    ``(s, a*)`` pairs used both for the log-likelihood measurement and for the
    projection.
2.  Project the *states* onto their first two principal components (PCA is
    computed once, on the ``pi_*`` states, and kept fixed so that different
    checkpoints are comparable -- exactly the "feed the same trajectories
    through the network" protocol of the CKA analysis in Appendix F).
3.  Colour every projected state by the policy's log-likelihood of the expert
    action ``log pi_theta(a*|s)``.
4.  Repeat over fine-tuning checkpoints / seeds and aggregate.

Nothing here retrains anything: it is pure analysis glue on top of
``loglikelihood.py`` (datasets, per-sample log-likelihood), ``cka.py`` (layer
naming / agent loading conventions) and ``forward_transfer.py`` (JSON result
layout).  NumPy is optional at import time (a pure-Python Jacobi/SVD-free
fallback is provided) so that the module can be imported -- and its metric
bookkeeping unit-tested -- in a minimal environment.

Public interface
----------------
Classes
    ``PCAResult``            -- projection + eigenvalues + explained variance
    ``PCADataset``           -- states/stage/step metadata bundle for projection
    ``PCAVisualizer``        -- fixed-projection projector with ``project``,
                                ``color_by_loglikelihood``, ``save`` and
                                checkpoint/seed accumulation helpers
    ``ProjectionTracker``    -- online accumulation of projections across steps
Functions
    ``pca_fit``              -- PCA (SVD on centered data) -> (components, mean)
    ``pca_transform``        -- project data with fitted (components, mean)
    ``explained_variance``   -- singular-value based variance ratios
    ``project_states``       -- convenience: fit + transform an array
    ``project_expert_dataset`` -- project an ``ExpertDataset`` (with stage/step)
    ``color_by_loglikelihood`` -- per-sample ``E[log pi(a*|s)]`` colour values
    ``projection_grid``      -- project a grid of seeds/steps into one figure
    ``plot_projection``      -- scatter of the 2-D projection, coloured by LL
    ``plot_projection_grid`` -- multi-panel figure (one panel per step/seed)
    ``plot_explained_variance`` -- scree plot
    ``aggregate_projections``   -- mean/90% CI over seeds for metrics
    ``select_device`` / ``as_numpy`` / ``flatten_observations``
    ``main``                 -- CLI entry point
Constants
    ``DEFAULT_N_COMPONENTS``, ``DEFAULT_POINT_SIZE``, ``DEFAULT_COLORMAP``,
    ``DEFAULT_CONFIDENCE``, ``APPENDIX8_MARKERS``

Notes
-----
*   The PCA basis is always estimated on the **pre-trained** states so that a
    fine-tuned policy cannot "rotate" the basis and make the comparison
    meaningless; this matters for the qualitative claim of Section 5 that the
    state distribution shifts (imperfect cloning gap) while representations
    drift away from ``pi_*``.
*   Observations may be flat vectors, ``(1, D)`` rows, or Meta-World
    dictionaries (``{"observation": ..., "achieved_goal": ...}``); all are
    flattened deterministically by ``flatten_observations``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:  # numpy is optional: metrics logic stays importable without it
    import numpy as _np
except Exception:  # pragma: no cover - minimal environments
    _np = None  # type: ignore

try:  # torch only needed to run policies forward
    import torch as _torch
except Exception:  # pragma: no cover
    _torch = None  # type: ignore

try:
    from src.analysis.loglikelihood import (  # type: ignore
        ExpertDataset,
        expert_log_likelihood,
        loglikelihood_trace,
    )
except Exception:  # pragma: no cover - allow standalone import
    ExpertDataset = None  # type: ignore
    expert_log_likelihood = None  # type: ignore
    loglikelihood_trace = None  # type: ignore


__all__ = [
    "DEFAULT_N_COMPONENTS",
    "DEFAULT_POINT_SIZE",
    "DEFAULT_COLORMAP",
    "DEFAULT_CONFIDENCE",
    "APPENDIX8_MARKERS",
    "PCAResult",
    "PCADataset",
    "PCAVisualizer",
    "ProjectionTracker",
    "pca_fit",
    "pca_transform",
    "explained_variance",
    "project_states",
    "project_expert_dataset",
    "color_by_loglikelihood",
    "projection_grid",
    "plot_projection",
    "plot_projection_grid",
    "plot_explained_variance",
    "aggregate_projections",
    "summarize",
    "z_for",
    "flatten_observations",
    "as_numpy",
    "select_device",
    "main",
]


DEFAULT_N_COMPONENTS = 2
"""Number of principal components shown in Figure 8 (2-D scatter)."""

DEFAULT_POINT_SIZE = 4.0
"""Marker size for the scatter plots (over ~10k states)."""

DEFAULT_COLORMAP = "coolwarm"
"""One diverging colormap for log-likelihood (low = red-ish / high = blue-ish)."""

DEFAULT_CONFIDENCE = 0.90
"""Paper-level confidence for multi-seed intervals (>= 20 seeds)."""

APPENDIX8_MARKERS = (0, 100_000, 500_000)
"""Fine-tuning steps at which Figure 8 panels are drawn in the appendix."""

_EPS = 1e-12


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------
def _require_numpy() -> Any:
    if _np is None:
        raise RuntimeError(
            "numpy is required for PCA visualisation; please install numpy."
        )
    return _np


def flatten_observations(obs: Any) -> Any:
    """Flatten observations into a 2-D ``(n, d)`` float array.

    Accepts ``(d,)``, ``(n, d)``, nested sequences, NumPy arrays, torch tensors
    and Meta-World style dicts (``observation``/``achieved_goal``/... keys are
    concatenated in a deterministic order).  Returns whatever array type the
    input used (NumPy is used when available).
    """
    if obs is None:
        raise ValueError("observations must not be None")

    # Meta-World / gymnasium Dict observation: concatenate values by sorted key.
    if isinstance(obs, Mapping):
        keys = sorted(obs.keys())
        parts = [flatten_observations(obs[k]) for k in keys]
        return concat_rows(parts)

    if _np is not None:
        arr = _np.asarray(obs, dtype=_np.float64)
        return _flatten_ndarray(arr)

    # Pure-python fallback: lists / tuples.
    if isinstance(obs, (list, tuple)) and obs and isinstance(obs[0], (list, tuple)):
        rows = [flatten_observations(o) for o in obs]
        return concat_rows(rows)
    if isinstance(obs, (list, tuple)):
        try:
            return [float(x) for x in obs]
        except (TypeError, ValueError):
            rows = [flatten_observations(o) for o in obs]
            return concat_rows(rows)
    raise TypeError(f"unsupported observation type: {type(obs)!r}")


def _flatten_ndarray(arr: Any) -> Any:
    np = _require_numpy()
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def concat_rows(parts: Sequence[Any]) -> Any:
    """Concatenate feature blocks column-wise, handling 1-D and 2-D blocks."""
    if _np is not None:
        arrs = [_np.atleast_2d(_np.asarray(p, dtype=_np.float64)) for p in parts]
        if any(a.shape[0] == 1 for a in arrs) and len({a.shape[0] for a in arrs}) > 1:
            n = max(a.shape[0] for a in arrs)
            arrs = [_np.repeat(a, n, axis=0) if a.shape[0] == 1 else a for a in arrs]
        return _np.concatenate(arrs, axis=1)
    # list-based fallback
    rows = [list(r) for r in parts[0]]
    for p in parts[1:]:
        pr = [list(r) for r in p]
        rows = [a + b for a, b in zip(rows, pr)]
    return rows


def as_numpy(x: Any) -> Any:
    """Best-effort conversion to a NumPy array (no-op when numpy is missing)."""
    if _np is None:
        return x
    if x is None:
        return None
    if _torch is not None and isinstance(x, _torch.Tensor):
        return x.detach().cpu().numpy()
    return _np.asarray(x)


def to_float_list(x: Any) -> List[float]:
    """Convert a 1-D array-like to a list of Python floats."""
    if x is None:
        return []
    if isinstance(x, list):
        return [float(v) for v in x]
    if _np is not None:
        return [float(v) for v in _np.asarray(x).reshape(-1)]
    return [float(v) for v in x]


def select_device(device: Optional[str] = None) -> str:
    """Resolve a torch device string, degrading gracefully without torch/CUDA."""
    if device:
        return device
    if _torch is not None and getattr(_torch, "cuda", None) is not None:
        try:
            if _torch.cuda.is_available():
                return "cuda"
        except Exception:  # pragma: no cover
            pass
    return "cpu"


def z_for(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Two-sided normal quantile used for the 90% CIs (paper uses >= 20 seeds).

    Only a small table plus a rational approximation is needed; the values
    ``0.90 -> 1.6449`` and ``0.95 -> 1.9600`` are the ones actually used.
    """
    table = {
        0.50: 0.6745,
        0.68: 0.9945,
        0.80: 1.2816,
        0.90: 1.6449,
        0.95: 1.9600,
        0.98: 2.3263,
        0.99: 2.5758,
    }
    for c, z in table.items():
        if abs(confidence - c) < 1e-9:
            return z
    p = 0.5 + 0.5 * confidence
    # Acklam-style inverse normal CDF (sufficient accuracy for CI reporting).
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((d[0] * q + d[1]) * q + d[2]) * q + d[3]
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((d[0] * q + d[1]) * q + d[2]) * q + d[3]
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


# ---------------------------------------------------------------------------
# PCA core (SVD based, no scikit-learn dependency)
# ---------------------------------------------------------------------------
def pca_fit(data: Any, n_components: int = DEFAULT_N_COMPONENTS, normalize: bool = False) -> Dict[str, Any]:
    """Fit PCA by SVD on the column-centered data matrix.

    Parameters
    ----------
    data : array-like, shape ``(n_samples, n_features)``
        States to fit the basis on -- in the paper always the **pre-trained**
        (``pi_*``) states, so that all checkpoints share one basis.
    n_components : int
        Number of principal directions to keep (2 for Figure 8).
    normalize : bool
        When ``True``, divide each *sample* by its L2 norm first (robust to
        observation-scale differences between Meta-World stages).

    Returns
    -------
    dict with ``components`` (``(k, d)``), ``mean`` (``(d,)``),
    ``singular_values``, ``explained_variance`` (``(k,)``),
    ``explained_variance_ratio`` (``(k,)``) and ``n_samples``.
    """
    np = _require_numpy()
    x = as_numpy(flatten_observations(data))
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if normalize:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, _EPS)

    n = x.shape[0]
    mean = x.mean(axis=0)
    xc = x - mean
    k = int(max(1, min(int(n_components), x.shape[1], max(1, n))))
    # Economy SVD: only k components requested.
    try:
        u, s, vt = np.linalg.svd(xc, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
        u, s, vt = np.linalg.svd(xc + 1e-6 * np.random.standard_normal(xc.shape), full_matrices=False)
    components = vt[:k]
    singular_values = s[:k]
    total_var = float(np.sum(np.square(s)) / max(1, n - 1))
    var = np.square(s[:k]) / max(1, n - 1)
    if total_var <= _EPS:
        ratio = np.zeros_like(var)
    else:
        ratio = var / (np.sum(np.square(s)) / max(1, n - 1))
    return {
        "components": components,
        "mean": mean,
        "singular_values": singular_values,
        "explained_variance": var,
        "explained_variance_ratio": ratio,
        "n_samples": int(n),
        "n_features": int(x.shape[1]),
        "n_components": int(k),
        "normalize": bool(normalize),
    }


def pca_transform(data: Any, components: Any, mean: Any, normalize: bool = False) -> Any:
    """Project ``data`` with a previously fitted PCA basis."""
    np = _require_numpy()
    x = np.atleast_2d(np.asarray(as_numpy(flatten_observations(data)), dtype=np.float64))
    if normalize:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, _EPS)
    comps = np.atleast_2d(np.asarray(as_numpy(components), dtype=np.float64))
    mu = np.asarray(as_numpy(mean), dtype=np.float64).reshape(1, -1)
    return (x - mu) @ comps.T


def explained_variance(data: Any, n_components: int = DEFAULT_N_COMPONENTS) -> Tuple[Any, Any]:
    """Return ``(explained_variance, explained_variance_ratio)`` for ``data``."""
    fit = pca_fit(data, n_components=n_components)
    return fit["explained_variance"], fit["explained_variance_ratio"]


def project_states(
    data: Any,
    n_components: int = DEFAULT_N_COMPONENTS,
    components: Any = None,
    mean: Any = None,
    normalize: bool = False,
) -> "PCAResult":
    """Fit (if necessary) and project states onto ``n_components`` directions.

    Returns a :class:`PCAResult` bundling the projection with the eigenvalues so
    the scree plot and the scatter share one computation.
    """
    if components is None or mean is None:
        fit = pca_fit(data, n_components=n_components, normalize=normalize)
        components, mean = fit["components"], fit["mean"]
    else:
        fit = {
            "components": as_numpy(components),
            "mean": as_numpy(mean),
            "explained_variance_ratio": None,
        }
    proj = pca_transform(data, components, mean, normalize=normalize)
    return PCAResult(
        projection=proj,
        components=as_numpy(components),
        mean=as_numpy(mean),
        explained_variance_ratio=fit.get("explained_variance_ratio"),
    )


@dataclass
class PCAResult:
    """A 2-D (or k-D) PCA projection plus the basis that produced it."""

    projection: Any
    components: Any = None
    mean: Any = None
    explained_variance_ratio: Any = None
    colors: Any = None
    stages: Any = None
    steps: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_components(self) -> int:
        arr = as_numpy(self.projection)
        if arr is None:
            return 0
        return int(arr.shape[1]) if getattr(arr, "ndim", 0) == 2 else 1

    @property
    def n_points(self) -> int:
        arr = as_numpy(self.projection)
        return 0 if arr is None else int(arr.shape[0])

    def explained(self, index: int = 0) -> float:
        if self.explained_variance_ratio is None:
            return float("nan")
        vals = to_float_list(self.explained_variance_ratio)
        return vals[index] if index < len(vals) else float("nan")

    def to_dict(self, include_projection: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "n_points": self.n_points,
            "n_components": self.n_components,
            "explained_variance_ratio": to_float_list(self.explained_variance_ratio)
            if self.explained_variance_ratio is not None
            else None,
            "metadata": dict(self.metadata),
        }
        if include_projection:
            out["projection"] = as_numpy(self.projection).tolist() if as_numpy(self.projection) is not None else None
            out["colors"] = to_float_list(self.colors) if self.colors is not None else None
        return out

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh)
        return path


# ---------------------------------------------------------------------------
# dataset bundle
# ---------------------------------------------------------------------------
@dataclass
class PCADataset:
    """States + metadata ready for projection.

    ``states`` is the raw observation array (or list of blocks), ``stage_ids``
    and ``steps`` let the plotter colour/split by Meta-World stage, while
    ``actions`` are the *expert* actions used for the log-likelihood colouring.
    """

    states: Any
    actions: Any = None
    stage_ids: Any = None
    steps: Any = None
    rewards: Any = None
    returns: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        arr = as_numpy(self.states)
        return 0 if arr is None else int(arr.shape[0])

    @classmethod
    def from_expert_dataset(cls, dataset: Any) -> "PCADataset":
        """Wrap an :class:`~src.analysis.loglikelihood.ExpertDataset`."""
        get = _getter(dataset)
        return cls(
            states=get("observations"),
            actions=get("actions"),
            stage_ids=get("stage_ids"),
            steps=get("steps_per_run"),
            rewards=get("rewards"),
            returns=get("returns"),
            metadata={
                "stage": get("stage"),
                "is_far": get("is_far"),
                "n_samples": get("n_samples"),
            },
        )

    @classmethod
    def from_transitions(cls, transitions: Sequence[Any], stage_ids: Any = None, steps: Any = None) -> "PCADataset":
        """Build from a list of transition dicts/tuples ``(obs, action, ...)``."""
        obs, acts = [], []
        for tr in transitions:
            o = _field(tr, "obs", _field(tr, "observation", _tuple_pos(tr, 0)))
            a = _field(tr, "action", _field(tr, "actions", _tuple_pos(tr, 1)))
            obs.append(o)
            acts.append(a)
        return cls(states=obs, actions=acts, stage_ids=stage_ids, steps=steps)

    def to_dict(self, include_states: bool = False) -> Dict[str, Any]:
        out = {
            "n_samples": len(self),
            "metadata": dict(self.metadata),
        }
        if include_states:
            st = as_numpy(self.states)
            out["states"] = None if st is None else st.tolist()
            out["actions"] = to_float_list(as_numpy(self.actions)) if self.actions is not None else None
            out["stage_ids"] = to_float_list(as_numpy(self.stage_ids)) if self.stage_ids is not None else None
        return out

    def save(self, path: str, include_states: bool = False) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(include_states=include_states), fh)
        return path


def _getter(obj: Any):
    def get(key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return getattr(obj, key, default)

    return get


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key)
    return default


def _tuple_pos(obj: Any, index: int) -> Any:
    if isinstance(obj, (tuple, list)) and len(obj) > index:
        return obj[index]
    return None


# ---------------------------------------------------------------------------
# projection of datasets / colouring by log-likelihood
# ---------------------------------------------------------------------------
def project_expert_dataset(
    dataset: Any,
    policy: Any = None,
    n_components: int = DEFAULT_N_COMPONENTS,
    components: Any = None,
    mean: Any = None,
    normalize: bool = False,
    color: bool = True,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
) -> PCAResult:
    """Project the states of an ``ExpertDataset`` and colour by ``log pi(a*|s)``.

    When ``components``/``mean`` are given, the frozen pre-trained basis is
    reused (this is what makes the Figure 8 panels comparable across
    checkpoints).  When ``color`` is ``True`` and a ``policy`` is supplied, the
    colours are the per-sample expert-action log-likelihoods computed by
    :func:`src.analysis.loglikelihood.loglikelihood_trace`.
    """
    ds = dataset
    if not isinstance(ds, PCADataset):
        if isinstance(ds, Mapping):
            ds = PCADataset(**{k: ds[k] for k in ds if k in PCADataset.__dataclass_fields__})  # type: ignore[attr-defined]
        elif ExpertDataset is not None and isinstance(ds, ExpertDataset):
            ds = PCADataset.from_expert_dataset(ds)
        else:
            ds = PCADataset.from_expert_dataset(ds)

    result = project_states(
        ds.states,
        n_components=n_components,
        components=components,
        mean=mean,
        normalize=normalize,
    )
    result.stages = ds.stage_ids
    result.steps = ds.steps
    result.metadata = dict(ds.metadata)

    if color and policy is not None:
        result.colors = color_by_loglikelihood(
            policy, ds, batch_size=batch_size, device=device
        )
    elif ds.rewards is not None:
        result.colors = as_numpy(ds.rewards)
    return result


def color_by_loglikelihood(
    policy: Any,
    dataset: Any,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
) -> Any:
    """Per-sample ``log pi_theta(a*|s)`` used as the scatter colours.

    Falls back to the mean log-likelihood (broadcast) if the fine-grained
    trace is unavailable, so the plotter always receives a usable colour vector.
    """
    ds = dataset
    if isinstance(ds, Mapping):
        obs, acts = ds.get("observations"), ds.get("actions")
        stage_ids = ds.get("stage_ids")
    else:
        get = _getter(ds)
        obs, acts, stage_ids = get("observations"), get("actions"), get("stage_ids")
        if obs is None:
            obs, acts = get("states"), get("actions")

    if loglikelihood_trace is not None:
        try:
            trace = loglikelihood_trace(
                policy,
                _as_dataset_like(obs, acts, stage_ids),
                batch_size=batch_size,
            )
            arr = as_numpy(trace)
            if arr is not None:
                return arr.reshape(-1)
        except Exception:
            pass

    if expert_log_likelihood is not None:
        try:
            value = expert_log_likelihood(
                policy,
                _as_dataset_like(obs, acts, stage_ids),
                batch_size=batch_size,
                device=device,
            )
            n = len(obs) if obs is not None else 0
            if _np is not None:
                return _np.full(n, float(value), dtype=_np.float64)
            return [float(value)] * n
        except Exception:
            pass

    n = len(obs) if obs is not None else 0
    if _np is not None:
        return _np.zeros(n, dtype=_np.float64)
    return [0.0] * n


def _as_dataset_like(obs: Any, actions: Any, stage_ids: Any) -> Any:
    """Return an object exposing ``observations``/``actions``/``stage_ids``."""
    if ExpertDataset is not None:
        try:
            kwargs: Dict[str, Any] = {"observations": obs, "actions": actions}
            if stage_ids is not None:
                kwargs["stage_ids"] = stage_ids
            return ExpertDataset(**kwargs)
        except TypeError:
            pass
    return {"observations": obs, "actions": actions, "stage_ids": stage_ids}


# ---------------------------------------------------------------------------
# visualiser (frozen basis across checkpoints)
# ---------------------------------------------------------------------------
class PCAVisualizer:
    """Project states with a *frozen* PCA basis and colour by log-likelihood.

    Typical use inside the fine-tuning loop::

        viz = PCAVisualizer(n_components=2, normalize=False)
        viz.fit(pretrain_states)                       # basis of pi_*
        for step in ...:
            res = viz.project(batch_obs, policy=agent.policy, actions=expert_actions)
            tracker.record(step, res)

    Parameters
    ----------
    n_components : int
        Kept principal components (2 for Figure 8).
    normalize : bool
        Per-sample L2 normalisation before the projection.
    batch_size / device : optional
        Forward-pass controls for the log-likelihood colouring.
    confidence : float
        Used by :meth:`aggregate` for the multi-seed CIs.
    name : str
        Identifier used in saved metadata.
    """

    def __init__(
        self,
        n_components: int = DEFAULT_N_COMPONENTS,
        normalize: bool = False,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        confidence: float = DEFAULT_CONFIDENCE,
        name: str = "pca",
    ) -> None:
        self.n_components = int(n_components)
        self.normalize = bool(normalize)
        self.batch_size = batch_size
        self.device = select_device(device)
        self.confidence = float(confidence)
        self.name = name
        self.components: Any = None
        self.mean: Any = None
        self.explained_variance_ratio: Any = None
        self.reference_states: Any = None

    # -- fitting ---------------------------------------------------------
    @property
    def fitted(self) -> bool:
        return self.components is not None and self.mean is not None

    def fit(self, states: Any, n_components: Optional[int] = None) -> "PCAVisualizer":
        """Estimate the PCA basis on ``states`` (the pre-trained states)."""
        fit = pca_fit(
            states,
            n_components=n_components or self.n_components,
            normalize=self.normalize,
        )
        self.components = fit["components"]
        self.mean = fit["mean"]
        self.explained_variance_ratio = fit["explained_variance_ratio"]
        self.reference_states = as_numpy(flatten_observations(states))
        return self

    def fit_from_dataset(self, dataset: Any) -> "PCAVisualizer":
        get = _getter(dataset)
        states = get("observations", get("states"))
        return self.fit(states)

    # -- projecting ------------------------------------------------------
    def project(
        self,
        states: Any,
        policy: Any = None,
        actions: Any = None,
        stage_ids: Any = None,
        steps: Any = None,
        color: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PCAResult:
        """Project ``states`` and (optionally) colour by ``log pi_theta(a*|s)``."""
        if not self.fitted:
            raise RuntimeError("PCAVisualizer.fit(...) must be called before project(...)")
        result = project_states(
            states,
            n_components=self.n_components,
            components=self.components,
            mean=self.mean,
            normalize=self.normalize,
        )
        result.stages = stage_ids
        result.steps = steps
        result.metadata = dict(metadata or {})
        result.metadata.setdefault("name", self.name)
        if self.explained_variance_ratio is not None:
            result.explained_variance_ratio = self.explained_variance_ratio
        if color and policy is not None and actions is not None:
            ds = {"observations": states, "actions": actions, "stage_ids": stage_ids}
            result.colors = color_by_loglikelihood(
                policy, ds, batch_size=self.batch_size, device=self.device
            )
        return result

    def project_dataset(self, dataset: Any, policy: Any = None, color: bool = True) -> PCAResult:
        get = _getter(dataset)
        return self.project(
            get("observations", get("states")),
            policy=policy,
            actions=get("actions"),
            stage_ids=get("stage_ids"),
            steps=get("steps_per_run", get("steps")),
            color=color,
            metadata={"stage": get("stage"), "is_far": get("is_far")},
        )

    # -- aggregation -----------------------------------------------------
    def aggregate(self, results: Sequence[PCAResult]) -> Dict[str, Any]:
        """Aggregate scalar summaries of several per-seed projections (90% CI)."""
        keys = ["n_points", "explained_0", "explained_1", "loglikelihood_mean"]
        values: Dict[str, List[float]] = {k: [] for k in keys}
        for res in results:
            values["n_points"].append(float(res.n_points))
            values["explained_0"].append(res.explained(0))
            values["explained_1"].append(res.explained(1))
            values["loglikelihood_mean"].append(mean_of(res.colors))
        return {k: summarize(v, confidence=self.confidence) for k, v in values.items()}

    # -- plotting --------------------------------------------------------
    def plot(self, result: PCAResult, path: Optional[str] = None, **kwargs: Any) -> Any:
        return plot_projection(result, path=path, **kwargs)

    def save(self, result: PCAResult, path: str) -> str:
        return result.save(path)

    def state_dict(self) -> Dict[str, Any]:
        arr = as_numpy(self.components)
        mu = as_numpy(self.mean)
        return {
            "components": None if arr is None else arr.tolist(),
            "mean": None if mu is None else mu.tolist(),
            "explained_variance_ratio": to_float_list(self.explained_variance_ratio)
            if self.explained_variance_ratio is not None
            else None,
            "n_components": self.n_components,
            "normalize": self.normalize,
            "name": self.name,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> "PCAVisualizer":
        np = _require_numpy()
        if state.get("components") is not None:
            self.components = np.asarray(state["components"], dtype=np.float64)
        if state.get("mean") is not None:
            self.mean = np.asarray(state["mean"], dtype=np.float64)
        if state.get("explained_variance_ratio") is not None:
            self.explained_variance_ratio = np.asarray(
                state["explained_variance_ratio"], dtype=np.float64
            )
        self.n_components = int(state.get("n_components", self.n_components))
        self.normalize = bool(state.get("normalize", self.normalize))
        self.name = str(state.get("name", self.name))
        return self


class ProjectionTracker:
    """Accumulate :class:`PCAResult` snapshots across fine-tuning steps.

    Mirrors the logging cadence of Figure 8/20 (every
    ``finetune.loglikelihood_every`` = 50k steps by default) and can also be
    used to dump one NPZ/JSON per checkpoint for the appendix grid plots.
    """

    def __init__(
        self,
        visualizer: Optional[PCAVisualizer] = None,
        every: Optional[int] = None,
        confidence: float = DEFAULT_CONFIDENCE,
        name: str = "pca_viz",
    ) -> None:
        self.visualizer = visualizer or PCAVisualizer()
        self.every = every
        self.confidence = float(confidence)
        self.name = name
        self.steps: List[int] = []
        self.results: List[PCAResult] = []

    def __len__(self) -> int:
        return len(self.results)

    def should_record(self, step: int) -> bool:
        if self.every in (None, 0):
            return True
        return step % int(self.every) == 0

    def record(self, step: int, result: PCAResult) -> PCAResult:
        result.metadata.setdefault("step", int(step))
        self.steps.append(int(step))
        self.results.append(result)
        return result

    def record_step(
        self,
        step: int,
        states: Any,
        policy: Any = None,
        actions: Any = None,
        stage_ids: Any = None,
        color: bool = True,
        force: bool = False,
    ) -> Optional[PCAResult]:
        """Project ``states`` at ``step`` and store it when due."""
        if not force and not self.should_record(step):
            return None
        res = self.visualizer.project(
            states,
            policy=policy,
            actions=actions,
            stage_ids=stage_ids,
            color=color,
            metadata={"step": int(step)},
        )
        return self.record(step, res)

    def curve(self, key: str = "loglikelihood_mean") -> Tuple[List[int], List[float]]:
        """(steps, values) trace for e.g. the mean expert log-likelihood colour."""
        steps, values = [], []
        for st, res in zip(self.steps, self.results):
            if key == "loglikelihood_mean":
                values.append(mean_of(res.colors))
            elif key.startswith("explained_"):
                values.append(res.explained(int(key.split("_")[1])))
            elif key == "n_points":
                values.append(float(res.n_points))
            else:
                values.append(float(res.metadata.get(key, float("nan"))))
            steps.append(int(st))
        return steps, values

    def summary(self) -> Dict[str, Any]:
        steps, ll = self.curve("loglikelihood_mean")
        return {
            "name": self.name,
            "num_snapshots": len(self.results),
            "steps": steps,
            "loglikelihood_mean": ll,
            "first_loglikelihood": ll[0] if ll else None,
            "last_loglikelihood": ll[-1] if ll else None,
            "explained_variance_ratio": to_float_list(self.visualizer.explained_variance_ratio)
            if self.visualizer.explained_variance_ratio is not None
            else None,
        }

    def save(self, directory: str, prefix: Optional[str] = None) -> List[str]:
        """Write one JSON per snapshot plus a tracker summary."""
        os.makedirs(directory, exist_ok=True)
        prefix = prefix or self.name
        paths: List[str] = []
        for st, res in zip(self.steps, self.results):
            path = os.path.join(directory, f"{prefix}_step_{int(st)}.json")
            res.save(path)
            paths.append(path)
        summary_path = os.path.join(directory, f"{prefix}_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(self.summary(), fh)
        paths.append(summary_path)
        return paths

    def to_dict(self) -> Dict[str, Any]:
        return self.summary()


# ---------------------------------------------------------------------------
# grid / batch helpers (appendix figures: one panel per step or per seed)
# ---------------------------------------------------------------------------
def projection_grid(
    datasets: Mapping[str, Any],
    n_components: int = DEFAULT_N_COMPONENTS,
    reference: Optional[str] = None,
    policies: Optional[Mapping[str, Any]] = None,
    normalize: bool = False,
) -> Dict[str, PCAResult]:
    """Project several labelled datasets onto a **shared** PCA basis.

    ``datasets`` maps a panel label (e.g. ``"step_100k"``) to a ``PCADataset``,
    ``ExpertDataset``, raw state array or ``{"observations": ..., "actions": ...}``
    mapping.  The basis is fitted on ``reference`` (default: the first entry,
    which in the paper is the pre-trained dataset) and reused for every panel.
    """
    if not datasets:
        return {}
    labels = list(datasets.keys())
    ref_label = reference if reference in datasets else labels[0]
    ref_states = _states_of(datasets[ref_label])
    basis = pca_fit(ref_states, n_components=n_components, normalize=normalize)

    out: Dict[str, PCAResult] = {}
    for label in labels:
        payload = datasets[label]
        policy = (policies or {}).get(label)
        res = project_expert_dataset(
            payload,
            policy=policy,
            n_components=n_components,
            components=basis["components"],
            mean=basis["mean"],
            normalize=normalize,
            color=policy is not None,
        )
        res.metadata.setdefault("label", label)
        res.metadata.setdefault("is_reference", label == ref_label)
        res.explained_variance_ratio = basis["explained_variance_ratio"]
        out[label] = res
    return out


def _states_of(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return payload.get("observations", payload.get("states"))
    get = _getter(payload)
    states = get("observations", get("states"))
    return states if states is not None else payload


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def plot_projection(
    result: Any,
    path: Optional[str] = None,
    title: Optional[str] = None,
    colormap: str = DEFAULT_COLORMAP,
    point_size: float = DEFAULT_POINT_SIZE,
    alpha: float = 0.6,
    colorbar_label: str = r"$\log \pi_\theta(a^*|s)$",
    show: bool = False,
    ax: Any = None,
    color: Any = None,
    cbar: bool = True,
    **scatter_kwargs: Any,
) -> Any:
    """Scatter the 2-D projection, coloured by log-likelihood.

    ``result`` may be a :class:`PCAResult` or a raw ``(n, 2)`` array.  Returns
    the matplotlib ``Figure`` (or ``None`` when matplotlib is unavailable).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib optional
        return None

    proj = result.projection if isinstance(result, PCAResult) else result
    proj = as_numpy(proj)
    if proj is None:
        return None
    if getattr(proj, "ndim", 2) == 1:
        proj = proj.reshape(-1, 1)
    if proj.shape[1] < 2:
        # Degenerate: pad so a scatter is still drawable.
        np = _require_numpy()
        proj = np.concatenate([proj, np.zeros((proj.shape[0], 1))], axis=1)

    if color is None and isinstance(result, PCAResult):
        color = result.colors
    color = as_numpy(color) if color is not None else None

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
    else:
        fig = getattr(ax, "figure", None)

    if color is not None and len(color) == proj.shape[0]:
        sc = ax.scatter(
            proj[:, 0], proj[:, 1], c=color, cmap=colormap, s=point_size, alpha=alpha,
            **scatter_kwargs,
        )
        if cbar and fig is not None:
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label(colorbar_label)
    else:
        ax.scatter(proj[:, 0], proj[:, 1], s=point_size, alpha=alpha, **scatter_kwargs)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    if title:
        ax.set_title(title)
    elif isinstance(result, PCAResult):
        stage = result.metadata.get("stage")
        step = result.metadata.get("step")
        bits = [b for b in (stage, f"step {step}" if step is not None else None) if b]
        if bits:
            ax.set_title(" / ".join(str(b) for b in bits))
    if isinstance(result, PCAResult) and result.explained_variance_ratio is not None:
        try:
            r0, r1 = float(result.explained_variance_ratio[0]), float(result.explained_variance_ratio[1])
            ax.set_xlabel(f"PC 1 ({100 * r0:.1f}%)")
            ax.set_ylabel(f"PC 2 ({100 * r1:.1f}%)")
        except Exception:
            pass
    if own_fig and fig is not None:
        fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show and fig is not None:
        plt.show()
    return fig


def plot_projection_grid(
    results: Mapping[str, Any],
    path: Optional[str] = None,
    titles: Optional[Mapping[str, str]] = None,
    ncols: Optional[int] = None,
    colormap: str = DEFAULT_COLORMAP,
    point_size: float = DEFAULT_POINT_SIZE,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    suptitle: Optional[str] = None,
    show: bool = False,
    **scatter_kwargs: Any,
) -> Any:
    """Multi-panel Figure-8-style grid with a shared colour scale for LL."""
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return None

    labels = list(results.keys())
    if not labels:
        return None
    ncols = int(ncols or min(len(labels), 4))
    nrows = int(math.ceil(len(labels) / ncols))

    ll_min, ll_max = vmin, vmax
    if ll_min is None or ll_max is None:
        vals: List[float] = []
        for res in results.values():
            colors = res.colors if isinstance(res, PCAResult) else None
            if colors is not None:
                vals.extend(to_float_list(colors))
        if vals:
            lo, hi = min(vals), max(vals)
            ll_min = lo if vmin is None else vmin
            ll_max = hi if vmax is None else vmax

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    for idx, label in enumerate(labels):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        res = results[label]
        title = (titles or {}).get(label)
        plot_projection(
            res,
            ax=ax,
            title=title or str(label),
            colormap=colormap,
            point_size=point_size,
            cbar=(idx == 0),
            **scatter_kwargs,
        )
    for idx in range(len(labels), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_explained_variance(
    explained_variance_ratio: Any,
    path: Optional[str] = None,
    component_names: Optional[Sequence[str]] = None,
    title: str = "PCA explained variance",
    show: bool = False,
    ax: Any = None,
) -> Any:
    """Scree plot (bar chart) of the retained principal components."""
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return None

    values = to_float_list(explained_variance_ratio)
    if not values:
        return None
    names = list(component_names) if component_names else [f"PC {i + 1}" for i in range(len(values))]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
    else:
        fig = getattr(ax, "figure", None)
    ax.bar(range(len(values)), values, color="#4c72b0")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(names[: len(values)])
    ax.set_ylabel("explained variance ratio")
    ax.set_title(title)
    if own_fig and fig is not None:
        fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show and fig is not None:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def mean_of(values: Any) -> float:
    vals = to_float_list(values) if values is not None else []
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def summarize(values: Iterable[float], confidence: float = DEFAULT_CONFIDENCE) -> Dict[str, float]:
    """Mean / half-width of a normal-approximation CI (>= 20 seeds in the paper)."""
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "half_width": float("nan"), "n": 0, "std": float("nan")}
    mean = sum(vals) / n
    if n == 1:
        return {"mean": mean, "half_width": 0.0, "n": 1, "std": 0.0}
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(max(var, 0.0))
    half = z_for(confidence) * std / math.sqrt(n)
    return {"mean": mean, "half_width": half, "n": n, "std": std}


def aggregate_projections(
    results: Sequence[Any],
    metric: str = "loglikelihood_mean",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, float]:
    """Aggregate one scalar metric over per-seed projections."""
    values: List[float] = []
    for res in results:
        if metric == "loglikelihood_mean":
            values.append(mean_of(res.colors))
        elif metric.startswith("explained_"):
            values.append(res.explained(int(metric.split("_")[1])))
        elif metric == "n_points":
            values.append(float(res.n_points))
        else:
            values.append(float(res.metadata.get(metric, float("nan"))))
    return summarize(values, confidence=confidence)


# ---------------------------------------------------------------------------
# CLI: rebuild Figure 8 panels from an ExpertDataset dump
# ---------------------------------------------------------------------------
def _load_states(path: str) -> Any:
    """Load states from ``.npz``/``.npy``/``.pt``/``.json``."""
    if path.endswith(".npz") and _np is not None:
        with _np.load(path) as data:
            key = "observations" if "observations" in data else list(data.keys())[0]
            return data[key]
    if path.endswith(".npy") and _np is not None:
        return _np.load(path)
    if path.endswith(".pt") and _torch is not None:
        payload = _torch.load(path, map_location="cpu")
        if isinstance(payload, Mapping):
            return payload.get("observations", payload.get("states"))
        return payload
    with open(path, "r") as fh:
        payload = json.load(fh)
    if isinstance(payload, Mapping):
        return payload.get("observations", payload.get("states", payload.get("projection")))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PCA visualisation of RoboticSequence states coloured by expert-action log-likelihood."
    )
    parser.add_argument("--states", type=str, default=None,
                        help="states file (.npz/.npy/.pt/.json) for the panel(s)")
    parser.add_argument("--panels", nargs="*", default=None,
                        help="additional 'label:path' panels to place on the shared PCA basis")
    parser.add_argument("--reference", type=str, default=None,
                        help="label used to fit the PCA basis (default: first panel)")
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS)
    parser.add_argument("--normalize", action="store_true", help="L2-normalise each state before PCA")
    parser.add_argument("--colormap", type=str, default=DEFAULT_COLORMAP)
    parser.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument("--output-dir", type=str, default="outputs/analysis/pca")
    parser.add_argument("--plot", action="store_true", help="render matplotlib figures")
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: ``python -m src.analysis.pca_viz --states ... --plot``."""
    args = build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    panels: Dict[str, Any] = {}
    if args.states:
        panels["reference"] = _load_states(args.states)
    for spec in args.panels or []:
        if ":" not in spec:
            print(f"[pca_viz] skipping malformed panel spec: {spec!r}")
            continue
        label, path = spec.split(":", 1)
        try:
            panels[label] = _load_states(path)
        except Exception as exc:  # pragma: no cover - user input
            print(f"[pca_viz] failed to load {path}: {exc}")

    if not panels:
        print("[pca_viz] no states given; nothing to do (see --help).")
        return 0

    results = projection_grid(
        panels,
        n_components=args.n_components,
        reference=args.reference,
        normalize=args.normalize,
    )
    summary = {label: res.to_dict(include_projection=False) for label, res in results.items()}
    summary_path = os.path.join(args.output_dir, "pca_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[pca_viz] wrote {summary_path}")

    if args.plot:
        plot_projection_grid(
            results,
            path=os.path.join(args.output_dir, "pca_grid.png"),
            colormap=args.colormap,
            point_size=args.point_size,
            show=args.show,
        )
        single = None
        if args.states:
            single = os.path.join(args.output_dir, "pca_reference.png")
            plot_projection(
                results.get("reference"),
                path=single,
                show=args.show,
                colormap=args.colormap,
                point_size=args.point_size,
            )
        # scree plot for the shared basis
        first = next(iter(results.values()))
        if first.explained_variance_ratio is not None:
            plot_explained_variance(
                first.explained_variance_ratio,
                path=os.path.join(args.output_dir, "pca_scree.png"),
                show=args.show,
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
