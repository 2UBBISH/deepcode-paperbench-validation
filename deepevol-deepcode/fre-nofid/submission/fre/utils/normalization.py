"""Observation / reward normalization utilities for FRE.

The FRE reproduction needs a small number of well-defined normalization maths:

* **Standardization** (a.k.a. "std-normalization"): ``(x - mean) / (std + eps)``.
  The paper applies this per-dimension to the *encoder* observations for the
  ExORL (walker / cheetah) domains, and the env wrappers / loaders refer to it.
* **Running statistics**: streaming mean/variance estimates (Welford-style)
  so that normalization statistics can be accumulated over a dataset or over
  training without materialising everything in memory twice.
* **Min/max scaling**: used when converting raw returns into the paper's
  normalized ``[0, 100]`` scores (see ``rewards/eval_rewards.py``).

Everything here is intentionally dependency-light (numpy only, torch optional)
so it can be imported from data loaders, env wrappers, reward functions and
training drivers alike.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:  # optional torch interop
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is a hard dependency for training
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


__all__ = [
    "EPS",
    "DEFAULT_CLIP",
    "Standardizer",
    "RunningMeanStd",
    "NormalizationStats",
    "compute_mean_std",
    "normalize",
    "denormalize",
    "standardize",
    "unstandardize",
    "normalize_states",
    "min_max_scale",
    "normalize_returns",
    "clip_normalized",
    "running_stats_from_batches",
    "load_stats",
    "save_stats",
    "stats_dict",
]


EPS = 1e-8
DEFAULT_CLIP = 10.0

ArrayLike = Union[np.ndarray, Sequence[float], float, int]


# --------------------------------------------------------------------------- #
# Core maths
# --------------------------------------------------------------------------- #
def compute_mean_std(
    x: ArrayLike,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = 0,
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, std)`` of ``x`` with a floor on the standard deviation.

    Parameters
    ----------
    x:
        Array-like values to summarize.
    axis:
        Axis (or axes) over which to reduce.  ``None`` reduces everything.
    eps:
        Minimum standard deviation; prevents division by ~0 for constant dims
        (this is the same epsilon used by the ExORL loader / wrappers).
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("compute_mean_std received an empty array")
    mean = np.mean(arr, axis=axis)
    std = np.std(arr, axis=axis)
    std = np.maximum(std, float(eps))
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(
    x: ArrayLike,
    mean: ArrayLike,
    std: ArrayLike,
    *,
    eps: float = EPS,
    clip: Optional[float] = None,
) -> np.ndarray:
    """Standardize ``x``: ``(x - mean) / (std + eps)``.

    Shapes broadcast, so a ``(D,)`` mean/std pair normalizes an ``(N, D)`` or
    ``(B, K, D)`` array.  Returns a ``float32`` array; if ``x`` was a torch
    tensor the caller should convert back (see :func:`to_tensor_like`).
    """
    arr = np.asarray(x, dtype=np.float32)
    m = np.asarray(mean, dtype=np.float32)
    s = np.asarray(std, dtype=np.float32)
    out = (arr - m) / (s + float(eps))
    if clip is not None:
        out = np.clip(out, -float(clip), float(clip))
    return out.astype(np.float32)


def denormalize(
    x: ArrayLike,
    mean: ArrayLike,
    std: ArrayLike,
    *,
    eps: float = EPS,
) -> np.ndarray:
    """Inverse of :func:`normalize`: ``x * (std + eps) + mean``."""
    arr = np.asarray(x, dtype=np.float32)
    m = np.asarray(mean, dtype=np.float32)
    s = np.asarray(std, dtype=np.float32)
    return (arr * (s + float(eps)) + m).astype(np.float32)


# Aliases used across the FRE code base / plan wording.
standardize = normalize
unstandardize = denormalize


def normalize_states(
    states: ArrayLike,
    mean: Optional[ArrayLike] = None,
    std: Optional[ArrayLike] = None,
    *,
    eps: float = 1e-3,
    clip: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize states, computing statistics when they are not provided.

    Returns ``(normalized_states, mean, std)`` so callers can persist the
    statistics for later evaluation time normalization.
    """
    arr = np.asarray(states, dtype=np.float32)
    if mean is None or std is None:
        mean, std = compute_mean_std(arr, axis=0, eps=eps)
    out = normalize(arr, mean, std, eps=eps, clip=clip)
    return out, np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def clip_normalized(x: ArrayLike, clip: float = DEFAULT_CLIP) -> np.ndarray:
    """Clip already-normalized values to ``[-clip, clip]``."""
    return np.clip(np.asarray(x, dtype=np.float32), -float(clip), float(clip))


def min_max_scale(
    x: ArrayLike,
    min_value: ArrayLike,
    max_value: ArrayLike,
    *,
    out_min: float = 0.0,
    out_max: float = 1.0,
    eps: float = EPS,
    clip: bool = True,
) -> np.ndarray:
    """Affine rescale ``x`` from ``[min_value, max_value]`` into ``[out_min, out_max]``."""
    arr = np.asarray(x, dtype=np.float32)
    lo = np.asarray(min_value, dtype=np.float32)
    hi = np.asarray(max_value, dtype=np.float32)
    scale = (float(out_max) - float(out_min)) / (hi - lo + float(eps))
    out = (arr - lo) * scale + float(out_min)
    if clip:
        out = np.clip(out, min(out_min, out_max), max(out_min, out_max))
    return out.astype(np.float32)


def normalize_returns(
    returns: ArrayLike,
    min_return: ArrayLike,
    max_return: ArrayLike,
    *,
    scale: float = 100.0,
    clip: bool = True,
) -> np.ndarray:
    """Map raw episode returns into the paper's normalized ``[0, scale]`` range.

    Table 1 in the FRE paper reports *normalized* returns in ``[0, 100]``; this
    helper performs exactly ``(r - r_min) / (r_max - r_min) * 100`` and clips
    to the valid range.
    """
    return min_max_scale(
        returns,
        min_return,
        max_return,
        out_min=0.0,
        out_max=float(scale),
        clip=clip,
    )


# --------------------------------------------------------------------------- #
# Running statistics
# --------------------------------------------------------------------------- #
class RunningMeanStd:
    """Streaming (Welford-style) mean / variance tracker.

    Keeps the count, mean and M2 (sum of squared deviations) for each element
    so that arbitrarily large datasets can be summarized in a single pass and
    merged later via :meth:`merge`.
    """

    def __init__(self, shape: Union[int, Sequence[int]] = (), eps: float = 1e-4):
        if isinstance(shape, (int, np.integer)):
            shape = (int(shape),)
        self.shape = tuple(int(d) for d in shape)
        self.eps = float(eps)
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(eps)

    # -- update ---------------------------------------------------------- #
    def update(self, x: ArrayLike) -> "RunningMeanStd":
        """Accumulate a batch of samples of shape ``(N,) + self.shape``."""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim > len(self.shape):
            arr = arr.reshape(-1, *self.shape)
        elif arr.ndim == 0 and self.shape:
            arr = arr.reshape(self.shape)
        batch_count = int(arr.shape[0]) if arr.ndim > 0 else 1
        if batch_count == 0:
            return self
        batch_mean = arr.mean(axis=0)
        batch_var = arr.var(axis=0)
        return self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> "RunningMeanStd":
        batch_mean = np.asarray(batch_mean, dtype=np.float64)
        batch_var = np.asarray(batch_var, dtype=np.float64)
        batch_count = float(batch_count)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 0.0)
        self.count = tot_count
        return self

    def merge(self, other: "RunningMeanStd") -> "RunningMeanStd":
        """Merge statistics of another tracker into ``self``."""
        if self.shape != other.shape:
            raise ValueError(
                f"shape mismatch merging RunningMeanStd: {self.shape} vs {other.shape}"
            )
        return self._update_from_moments(other.mean, other.var, other.count)

    # -- accessors ------------------------------------------------------- #
    @property
    def std(self) -> np.ndarray:
        return np.maximum(np.sqrt(self.var), self.eps)

    @property
    def n(self) -> float:
        return self.count

    def normalize(self, x: ArrayLike, clip: Optional[float] = None) -> np.ndarray:
        return normalize(x, self.mean, self.std, eps=0.0, clip=clip)

    def denormalize(self, x: ArrayLike) -> np.ndarray:
        return denormalize(x, self.mean, self.std, eps=0.0)

    # -- serialization --------------------------------------------------- #
    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
            "shape": list(self.shape),
            "eps": float(self.eps),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> "RunningMeanStd":
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state.get("count", self.eps))
        if "shape" in state and tuple(state["shape"]) != self.shape:
            self.shape = tuple(int(d) for d in state["shape"])
        if "eps" in state:
            self.eps = float(state["eps"])
        return self

    def copy(self) -> "RunningMeanStd":
        other = RunningMeanStd(self.shape, eps=self.eps)
        other.mean = self.mean.copy()
        other.var = self.var.copy()
        other.count = self.count
        return other

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"RunningMeanStd(shape={self.shape}, count={self.count:.1f}, "
            f"mean={np.asarray(self.mean).ravel()[:4]}, std={np.asarray(self.std).ravel()[:4]})"
        )


def running_stats_from_batches(
    batches,
    *,
    eps: float = 1e-3,
    clip: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ``(mean, std)`` over an iterable of observation batches.

    ``batches`` may be numpy arrays, torch tensors, or dicts containing an
    ``"observations"`` key (the layout produced by the dataloaders).
    """
    tracker: Optional[RunningMeanStd] = None
    for batch in batches:
        arr = _as_observation_array(batch)
        if arr is None or arr.size == 0:
            continue
        if tracker is None:
            tracker = RunningMeanStd(shape=arr.shape[1:], eps=eps)
        tracker.update(arr)
    if tracker is None:
        raise ValueError("running_stats_from_batches received no observations")
    mean = tracker.mean.astype(np.float32)
    std = np.maximum(tracker.std, float(eps)).astype(np.float32)
    return mean, std


def _as_observation_array(batch) -> Optional[np.ndarray]:
    """Best-effort extraction of an ``(N, ...)`` observation array."""
    if batch is None:
        return None
    if isinstance(batch, Mapping):
        for key in ("observations", "observation", "obs", "encoder_observations", "states"):
            if key in batch:
                return _as_observation_array(batch[key])
        return None
    if _TORCH_AVAILABLE and isinstance(batch, torch.Tensor):  # type: ignore[arg-type]
        return batch.detach().cpu().numpy()
    arr = np.asarray(batch)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Convenience wrappers
# --------------------------------------------------------------------------- #
@dataclass
class NormalizationStats:
    """Container binding ``mean``/``std`` with the eps used to compute them."""

    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-3
    clip: Optional[float] = None

    def transform(self, x: ArrayLike) -> np.ndarray:
        return normalize(x, self.mean, self.std, eps=self.eps, clip=self.clip)

    def inverse(self, x: ArrayLike) -> np.ndarray:
        return denormalize(x, self.mean, self.std, eps=self.eps)

    @property
    def dim(self) -> int:
        return int(np.asarray(self.mean).shape[-1]) if np.asarray(self.mean).ndim else 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mean": np.asarray(self.mean, dtype=np.float32).tolist(),
            "std": np.asarray(self.std, dtype=np.float32).tolist(),
            "eps": float(self.eps),
            "clip": None if self.clip is None else float(self.clip),
        }

    @classmethod
    def from_array(cls, x: ArrayLike, eps: float = 1e-3, clip: Optional[float] = None):
        mean, std = compute_mean_std(x, axis=0, eps=eps)
        return cls(mean=mean, std=std, eps=eps, clip=clip)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            eps=float(data.get("eps", 1e-3)),
            clip=None if data.get("clip") is None else float(data["clip"]),
        )


class Standardizer:
    """Fit/transform observation standardizer with optional clipping.

    Mirrors the behaviour of the ExORL preprocessing step (per-dimension
    std-normalization of the encoder observations) while being usable as a
    standalone object::

        std = Standardizer(eps=1e-3, clip=10.0)
        std.fit(observations)
        obs_n = std.transform(observations)
    """

    def __init__(
        self,
        mean: Optional[ArrayLike] = None,
        std: Optional[ArrayLike] = None,
        *,
        eps: float = 1e-3,
        clip: Optional[float] = None,
    ):
        self.eps = float(eps)
        self.clip = None if clip is None else float(clip)
        self.mean: Optional[np.ndarray] = (
            None if mean is None else np.asarray(mean, dtype=np.float32)
        )
        self.std: Optional[np.ndarray] = (
            None if std is None else np.asarray(std, dtype=np.float32)
        )

    # -- api ------------------------------------------------------------- #
    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def fit(self, x: ArrayLike) -> "Standardizer":
        mean, std = compute_mean_std(x, axis=0, eps=self.eps)
        self.mean, self.std = mean, std
        return self

    def fit_transform(self, x: ArrayLike) -> np.ndarray:
        return self.fit(x).transform(x)

    def transform(self, x: ArrayLike) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Standardizer.transform called before fit")
        return normalize(x, self.mean, self.std, eps=self.eps, clip=self.clip)

    def inverse_transform(self, x: ArrayLike) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Standardizer.inverse_transform called before fit")
        return denormalize(x, self.mean, self.std, eps=self.eps)

    # aliases matching the loader / wrapper vocabulary
    normalize = transform
    denormalize = inverse_transform

    # -- conversion ------------------------------------------------------ #
    def to_stats(self) -> NormalizationStats:
        if not self.fitted:
            raise RuntimeError("Standardizer is not fitted")
        return NormalizationStats(self.mean, self.std, eps=self.eps, clip=self.clip)

    @classmethod
    def from_stats(cls, stats: Union[NormalizationStats, Mapping[str, Any]]) -> "Standardizer":
        if isinstance(stats, NormalizationStats):
            return cls(stats.mean, stats.std, eps=stats.eps, clip=stats.clip)
        return cls(
            np.asarray(stats["mean"], dtype=np.float32),
            np.asarray(stats["std"], dtype=np.float32),
            eps=float(stats.get("eps", 1e-3)),
            clip=None if stats.get("clip") is None else float(stats["clip"]),
        )

    def state_dict(self) -> Dict[str, Any]:
        return self.to_stats().as_dict() if self.fitted else {"mean": None, "std": None}

    def load_state_dict(self, state: Mapping[str, Any]) -> "Standardizer":
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.std = np.asarray(state["std"], dtype=np.float32)
        self.eps = float(state.get("eps", self.eps))
        clip = state.get("clip", self.clip)
        self.clip = None if clip is None else float(clip)
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Standardizer(fitted={self.fitted}, eps={self.eps}, clip={self.clip}, "
            f"dim={0 if self.mean is None else np.asarray(self.mean).shape[-1]})"
        )


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #
def stats_dict(mean: ArrayLike, std: ArrayLike, eps: float = 1e-3) -> Dict[str, Any]:
    """Bundle ``(mean, std)`` into a JSON-friendly dict."""
    return {
        "mean": np.asarray(mean, dtype=np.float32).tolist(),
        "std": np.asarray(std, dtype=np.float32).tolist(),
        "eps": float(eps),
    }


def save_stats(path: str, mean: ArrayLike, std: ArrayLike, eps: float = 1e-3) -> str:
    """Write normalization statistics to a JSON file (creating directories)."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = stats_dict(mean, std, eps=eps)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_stats(path: str) -> NormalizationStats:
    """Read normalization statistics previously written by :func:`save_stats`."""
    with open(path, "r") as fh:
        payload = json.load(fh)
    return NormalizationStats.from_dict(payload)


# --------------------------------------------------------------------------- #
# torch interop
# --------------------------------------------------------------------------- #
def to_tensor_like(original, array: np.ndarray):
    """Convert a numpy result back to the type/device of ``original``.

    Used so normalization is transparent when a tensor is passed in (the env
    wrappers and policy nets may feed either numpy or torch).
    """
    if _TORCH_AVAILABLE and isinstance(original, torch.Tensor):  # type: ignore[arg-type]
        return torch.as_tensor(array, dtype=original.dtype, device=original.device)
    return array


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    rng = np.random.default_rng(0)
    data = rng.normal(loc=3.0, scale=2.0, size=(1000, 5)).astype(np.float32)

    stats = Standardizer().fit(data)
    normed = stats.transform(data)
    assert np.allclose(normed.mean(axis=0), 0.0, atol=1e-2), normed.mean(axis=0)
    assert np.allclose(normed.std(axis=0), 1.0, atol=1e-2), normed.std(axis=0)
    assert np.allclose(stats.inverse_transform(normed), data, atol=1e-4)

    tracker = RunningMeanStd(shape=(5,))
    tracker.update(data[:500])
    tracker.update(data[500:])
    assert np.allclose(tracker.mean, data.mean(axis=0), atol=1e-6)
    assert np.allclose(tracker.std, data.std(axis=0), atol=1e-6)

    mean, std = running_stats_from_batches([{"observations": data[:300]}, data[300:]])
    assert np.allclose(mean, data.mean(axis=0), atol=1e-6)
    assert np.allclose(std, data.std(axis=0), atol=1e-6)

    scores = normalize_returns(np.array([-2000.0, -1000.0, 0.0]), -2000.0, 0.0)
    assert np.allclose(scores, [0.0, 50.0, 100.0]), scores

    print("normalization.py self-test passed")
