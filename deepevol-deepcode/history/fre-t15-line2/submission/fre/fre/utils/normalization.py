"""Dataset statistics and per-dimension state normalization for FRE.

Paper reference (Appendix C.2, ExORL):

    "Goals in ExORL are computed when the Euclidean distance between the current
    state and the goal state is less than 0.1. Each state dimension is normalized
    according to the standard deviation along that dimension within the offline
    dataset. Augmented information is not utilized when calculating goal distance."

So the canonical statistic used throughout this codebase is the *per-dimension
standard deviation computed over the offline dataset*.  We additionally store the
per-dimension mean (subtracting the mean is the standard practice and the paper is
silent on whether it is subtracted; both behaviours are supported via the
``subtract_mean`` flag, whose default is ``True``).  Goal distances for ExORL use
``mean=None``-style pure-std scaling through :func:`normalize_by_std` with
``subtract_mean=False`` -- see ``fre/envs/exorl_tasks.py``, which passes the
dataset std through ``state_std`` and normalizes with the paper's convention.

The module provides:

* :class:`RunningMeanStd` -- online (streaming) mean/variance estimator that can
  consume datasets incrementally without materialising them twice.
* :class:`DatasetStatistics` -- immutable-ish container holding mean/std/mins/maxs
  for one observation space, with normalize/denormalize helpers and persistence.
* :func:`normalize_by_std` / :func:`unnormalize_by_std` -- functional per-dimension
  normalization used by the dataset loaders and evaluation harness.
* :func:`compute_statistics` / :func:`dataset_statistics_from_array` -- one-shot
  statistics estimators from numpy arrays.
* :func:`normalize_observations` / :func:`unnormalize_observations` -- thin wrappers
  kept name-compatible with :mod:`fre.data.exorl_dataset`.

All statistics floor the standard deviation at ``STD_FLOOR = 1e-6`` (constant or
near-constant dimensions must not blow up).  Means are optional so that the ExORL
goal-distance convention (std only) is directly expressible.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "STD_FLOOR",
    "RunningMeanStd",
    "DatasetStatistics",
    "normalize_by_std",
    "unnormalize_by_std",
    "normalize_observations",
    "unnormalize_observations",
    "compute_statistics",
    "compute_mean_std",
    "compute_std",
    "dataset_statistics_from_array",
    "apply_normalization",
]

#: Floor applied to per-dimension standard deviations (matches the loaders).
STD_FLOOR = 1e-6

ArrayLike = Union[np.ndarray, Sequence[float], float]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _as_2d_float(values: ArrayLike) -> np.ndarray:
    """Coerce ``values`` to a 2-D float64 numpy array (rows = samples)."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def _as_1d_float(values: Optional[ArrayLike], dim: Optional[int] = None) -> Optional[np.ndarray]:
    """Coerce ``values`` to a 1-D float array (or ``None``)."""
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if dim is not None and array.size != dim:
        raise ValueError(f"Expected a vector of dimension {dim}, got {array.shape}")
    return array


def _safe_std(std: np.ndarray, floor: float = STD_FLOOR) -> np.ndarray:
    """Replace non-finite / below-floor standard deviations with ``floor``."""
    std = np.asarray(std, dtype=np.float64)
    std = np.where(np.isfinite(std), std, 0.0)
    return np.maximum(std, floor)


# ---------------------------------------------------------------------------
# Running statistics (streaming)
# ---------------------------------------------------------------------------
@dataclass
class RunningMeanStd:
    """Online mean / variance estimator (parallel (Chan et al.) update rule).

    Parameters
    ----------
    shape:
        Per-dimension shape of a single observation (``(obs_dim,)`` or ``()``).
    epsilon:
        Floor applied to the running variance to avoid division by zero.
    mean, var, count:
        Optional warm-start values (e.g. restored from a checkpoint).
    """

    shape: Tuple[int, ...] = ()
    epsilon: float = 1e-4
    mean: Optional[np.ndarray] = None
    var: Optional[np.ndarray] = None
    count: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.shape, int):
            self.shape = (int(self.shape),)
        self.shape = tuple(int(s) for s in self.shape)
        if self.mean is None:
            self.mean = np.zeros(self.shape, dtype=np.float64)
        else:
            self.mean = np.asarray(self.mean, dtype=np.float64).reshape(self.shape)
        if self.var is None:
            self.var = np.ones(self.shape, dtype=np.float64)
        else:
            self.var = np.asarray(self.var, dtype=np.float64).reshape(self.shape)
        self.count = float(self.count)

    # -- updates ---------------------------------------------------------
    def update(self, values: ArrayLike) -> "RunningMeanStd":
        """Fuse a batch of samples ``values`` into the running statistics."""
        batch = np.asarray(values, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        if batch.ndim != 2:
            batch = batch.reshape(-1, int(np.prod(self.shape)))
        batch = batch.reshape(batch.shape[0], -1)
        if batch.size == 0:
            return self
        if self.mean is None or self.mean.size != batch.shape[1]:
            self.shape = (batch.shape[1],)
            self.mean = np.zeros(self.shape, dtype=np.float64)
            self.var = np.ones(self.shape, dtype=np.float64)
            self.count = 0.0

        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = float(batch.shape[0])
        self._update_from_moments(batch_mean, batch_var, batch_count)
        return self

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float
    ) -> None:
        """Parallel-variance update with a new batch's mean/var/count."""
        if self.count == 0.0:
            self.mean = np.asarray(batch_mean, dtype=np.float64).copy()
            self.var = np.asarray(batch_var, dtype=np.float64).copy()
            self.count = batch_count
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = np.asarray(batch_var, dtype=np.float64) * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    # -- moments ---------------------------------------------------------
    @property
    def std(self) -> np.ndarray:
        """Per-dimension standard deviation, floored at ``epsilon``."""
        return np.sqrt(np.maximum(self.var, self.epsilon))

    @property
    def variance(self) -> np.ndarray:
        return np.maximum(np.asarray(self.var, dtype=np.float64), self.epsilon)

    def reset(self) -> "RunningMeanStd":
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = 0.0
        return self

    # -- normalization ---------------------------------------------------
    def normalize(self, values: ArrayLike, subtract_mean: bool = True, clip: Optional[float] = None):
        """Standardize ``values`` using the running moments."""
        array = np.asarray(values, dtype=np.float64)
        centered = array - self.mean if subtract_mean else array
        out = centered / self.std
        if clip is not None:
            out = np.clip(out, -abs(float(clip)), abs(float(clip)))
        return out

    def denormalize(self, values: ArrayLike, subtract_mean: bool = True):
        array = np.asarray(values, dtype=np.float64)
        out = array * self.std
        if subtract_mean:
            out = out + self.mean
        return out

    # -- (de)serialization ----------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "shape": list(self.shape),
            "epsilon": self.epsilon,
            "mean": np.asarray(self.mean).tolist(),
            "var": np.asarray(self.var).tolist(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "RunningMeanStd":
        self.shape = tuple(state.get("shape", np.asarray(state["mean"]).shape))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state.get("count", 0.0))
        return self

    def to_dataset_statistics(self, subtract_mean: bool = True) -> "DatasetStatistics":
        return DatasetStatistics(
            mean=np.asarray(self.mean, dtype=np.float64) if subtract_mean else None,
            std=self.std,
            count=float(self.count),
            subtract_mean=subtract_mean,
        )


# ---------------------------------------------------------------------------
# Dataset statistics container
# ---------------------------------------------------------------------------
@dataclass
class DatasetStatistics:
    """Per-dimension statistics of an observation space.

    Parameters
    ----------
    mean:
        Per-dimension mean (``None`` -> no mean subtraction, std-only scaling).
    std:
        Per-dimension standard deviation.  Non-finite values and values below
        ``std_floor`` are replaced by ``std_floor``.
    mins, maxs:
        Optional per-dimension minima/maxima (used for state boxes).
    count:
        Number of samples the statistics were computed from.
    std_floor:
        Lower bound applied to ``std`` (default :data:`STD_FLOOR`).
    subtract_mean:
        Whether :meth:`normalize` subtracts ``mean`` (default ``True``).
    name:
        Free-form label (e.g. ``"walker"`` / ``"antmaze-large-diverse-v2"``).
    """

    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    mins: Optional[np.ndarray] = None
    maxs: Optional[np.ndarray] = None
    count: float = 0.0
    std_floor: float = STD_FLOOR
    subtract_mean: bool = True
    name: str = "dataset"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean = _as_1d_float(self.mean)
        self.mins = _as_1d_float(self.mins)
        self.maxs = _as_1d_float(self.maxs)
        if self.std is None:
            raise ValueError("DatasetStatistics requires a per-dimension `std`")
        self.std = _safe_std(_as_1d_float(self.std), self.std_floor)
        if self.mean is not None and self.mean.size != self.std.size:
            raise ValueError(
                f"`mean` (dim {self.mean.size}) and `std` (dim {self.std.size}) must match"
            )
        for name, vec in (("mins", self.mins), ("maxs", self.maxs)):
            if vec is not None and vec.size != self.std.size:
                raise ValueError(f"`{name}` (dim {vec.size}) and `std` (dim {self.std.size}) must match")
        self.count = float(self.count)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_data(
        cls,
        observations: ArrayLike,
        std_floor: float = STD_FLOOR,
        subtract_mean: bool = True,
        name: str = "dataset",
        weighted_stats: Optional[Dict[str, float]] = None,
    ) -> "DatasetStatistics":
        """Compute per-dimension statistics directly from an array of states."""
        data = _as_2d_float(observations)
        return cls.from_moments(
            mean=data.mean(axis=0),
            var=data.var(axis=0),
            mins=data.min(axis=0),
            maxs=data.max(axis=0),
            count=float(data.shape[0]),
            std_floor=std_floor,
            subtract_mean=subtract_mean,
            name=name,
        )

    @classmethod
    def from_moments(
        cls,
        mean: Optional[ArrayLike] = None,
        var: Optional[ArrayLike] = None,
        std: Optional[ArrayLike] = None,
        mins: Optional[ArrayLike] = None,
        maxs: Optional[ArrayLike] = None,
        count: float = 0.0,
        std_floor: float = STD_FLOOR,
        subtract_mean: bool = True,
        name: str = "dataset",
    ) -> "DatasetStatistics":
        """Build statistics from pre-computed moments (mean/var *or* std)."""
        if std is None:
            if var is None:
                raise ValueError("Provide either `std` or `var`")
            std = np.sqrt(np.maximum(np.asarray(var, dtype=np.float64), 0.0))
        return cls(
            mean=mean,
            std=std,
            mins=mins,
            maxs=maxs,
            count=count,
            std_floor=std_floor,
            subtract_mean=subtract_mean,
            name=name,
        )

    # -- properties ------------------------------------------------------
    @property
    def dim(self) -> int:
        return int(np.asarray(self.std).size)

    @property
    def variance(self) -> np.ndarray:
        return np.square(self.std)

    def __len__(self) -> int:  # pragma: no cover - convenience
        return self.dim

    # -- normalization ---------------------------------------------------
    def normalize(
        self,
        observations: ArrayLike,
        subtract_mean: Optional[bool] = None,
        clip: Optional[float] = None,
        in_place: bool = False,
    ) -> np.ndarray:
        """Per-dimension standardization (std scaling, optional mean shift)."""
        return normalize_by_std(
            observations,
            std=self.std,
            mean=self.mean,
            subtract_mean=self.subtract_mean if subtract_mean is None else subtract_mean,
            clip=clip,
            in_place=in_place,
        )

    def denormalize(
        self,
        observations: ArrayLike,
        subtract_mean: Optional[bool] = None,
        in_place: bool = False,
    ) -> np.ndarray:
        """Inverse of :meth:`normalize`."""
        return unnormalize_by_std(
            observations,
            std=self.std,
            mean=self.mean,
            subtract_mean=self.subtract_mean if subtract_mean is None else subtract_mean,
            in_place=in_place,
        )

    # -- misc ------------------------------------------------------------
    def state_box(self, num_std: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Return a ``(low, high)`` state box.

        Uses recorded minima/maxima when available, otherwise falls back to
        ``mean +/- num_std * std`` around the dataset mean (a documented default;
        the paper does not specify how bounded state ranges are obtained).
        """
        if self.mins is not None and self.maxs is not None:
            return np.asarray(self.mins, dtype=np.float64), np.asarray(self.maxs, dtype=np.float64)
        if self.mean is None:
            half = num_std * np.asarray(self.std, dtype=np.float64)
            return -half, half
        return self.mean - num_std * self.std, self.mean + num_std * self.std

    def copy(self, **overrides: Any) -> "DatasetStatistics":
        values = dict(
            mean=None if self.mean is None else self.mean.copy(),
            std=self.std.copy(),
            mins=None if self.mins is None else self.mins.copy(),
            maxs=None if self.maxs is None else self.maxs.copy(),
            count=self.count,
            std_floor=self.std_floor,
            subtract_mean=self.subtract_mean,
            name=self.name,
            metadata=dict(self.metadata),
        )
        values.update(overrides)
        return DatasetStatistics(**values)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "count": self.count,
            "std_floor": self.std_floor,
            "subtract_mean": self.subtract_mean,
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": self.std.tolist(),
            "mins": None if self.mins is None else self.mins.tolist(),
            "maxs": None if self.maxs is None else self.maxs.tolist(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "DatasetStatistics":
        values = dict(values)
        values.pop("dim", None)
        return cls(**values)

    def save(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "DatasetStatistics":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def describe(self) -> Dict[str, Any]:
        values = self.to_dict()
        values["mean_abs"] = None if self.mean is None else float(np.mean(np.abs(self.mean)))
        values["std_min"] = float(np.min(self.std))
        values["std_max"] = float(np.max(self.std))
        return values


# ---------------------------------------------------------------------------
# Functional normalization API
# ---------------------------------------------------------------------------
def normalize_by_std(
    observations: ArrayLike,
    std: ArrayLike,
    mean: Optional[ArrayLike] = None,
    subtract_mean: bool = True,
    clip: Optional[float] = None,
    in_place: bool = False,
    std_floor: float = STD_FLOOR,
) -> np.ndarray:
    """Normalize ``observations`` by the per-dimension standard deviation.

    Implements the paper's ExORL convention ("Each state dimension is normalized
    according to the standard deviation along that dimension within the offline
    dataset").  ``mean`` is optional: when omitted (or ``subtract_mean=False``)
    only the std scaling is applied, which is exactly the transform used for ExORL
    goal distances.

    Parameters
    ----------
    observations:
        Array of shape ``(..., state_dim)``.
    std:
        Per-dimension standard deviations from the offline dataset.
    mean:
        Per-dimension means; ignored when ``subtract_mean`` is ``False``.
    subtract_mean:
        Whether to center the observations first (default ``True``).
    clip:
        Optional symmetric clipping applied after normalization.
    in_place:
        When ``True`` and ``observations`` is a float array, modify it in place.
    std_floor:
        Lower bound applied to ``std`` before dividing.
    """
    array = np.asarray(observations)
    if in_place and isinstance(observations, np.ndarray) and array.dtype.kind == "f":
        out = observations
    else:
        out = np.array(array, dtype=np.float64, copy=True)

    std_vec = _safe_std(np.asarray(std, dtype=np.float64).reshape(-1), std_floor)
    if clip is not None:
        limit = abs(float(clip))
        np.clip(out, -limit, limit, out=out) if False else None  # normalization first
    if subtract_mean and mean is not None:
        mean_vec = _as_1d_float(mean, std_vec.size)
        if mean_vec is not None:
            out -= mean_vec
    out /= std_vec
    if clip is not None:
        np.clip(out, -abs(float(clip)), abs(float(clip)), out=out)
    return out


def unnormalize_by_std(
    observations: ArrayLike,
    std: ArrayLike,
    mean: Optional[ArrayLike] = None,
    subtract_mean: bool = True,
    in_place: bool = False,
    std_floor: float = STD_FLOOR,
) -> np.ndarray:
    """Inverse of :func:`normalize_by_std`."""
    array = np.asarray(observations)
    if in_place and isinstance(observations, np.ndarray) and array.dtype.kind == "f":
        out = observations
    else:
        out = np.array(array, dtype=np.float64, copy=True)

    std_vec = _safe_std(np.asarray(std, dtype=np.float64).reshape(-1), std_floor)
    out *= std_vec
    if subtract_mean and mean is not None:
        mean_vec = _as_1d_float(mean, std_vec.size)
        if mean_vec is not None:
            out += mean_vec
    return out


def normalize_observations(
    observations: ArrayLike,
    mean: Optional[ArrayLike] = None,
    std: Optional[ArrayLike] = None,
    stats: Optional[DatasetStatistics] = None,
    in_place: bool = False,
    std_floor: float = STD_FLOOR,
) -> np.ndarray:
    """Name-compatible wrapper around :func:`normalize_by_std`."""
    if stats is not None:
        mean, std = stats.mean, stats.std
    if std is None:
        raise ValueError("`std` (or `stats`) must be provided for normalization")
    return normalize_by_std(
        observations,
        std=std,
        mean=mean,
        subtract_mean=mean is not None,
        in_place=in_place,
        std_floor=std_floor,
    )


def unnormalize_observations(
    observations: ArrayLike,
    mean: Optional[ArrayLike] = None,
    std: Optional[ArrayLike] = None,
    stats: Optional[DatasetStatistics] = None,
    in_place: bool = False,
    std_floor: float = STD_FLOOR,
) -> np.ndarray:
    """Name-compatible wrapper around :func:`unnormalize_by_std`."""
    if stats is not None:
        mean, std = stats.mean, stats.std
    if std is None:
        raise ValueError("`std` (or `stats`) must be provided for unnormalization")
    return unnormalize_by_std(
        observations,
        std=std,
        mean=mean,
        subtract_mean=mean is not None,
        in_place=in_place,
        std_floor=std_floor,
    )


def apply_normalization(
    observations: ArrayLike,
    statistics: Optional[Union[DatasetStatistics, Dict[str, Any]]],
    in_place: bool = False,
) -> np.ndarray:
    """Normalize with a :class:`DatasetStatistics` (or raw dict), or a no-op if ``None``."""
    if statistics is None:
        return np.asarray(observations, dtype=np.float64)
    if isinstance(statistics, dict):
        statistics = DatasetStatistics.from_dict(statistics)
    return statistics.normalize(observations, in_place=in_place)


# ---------------------------------------------------------------------------
# Statistics estimation helpers
# ---------------------------------------------------------------------------
def compute_statistics(
    observations: ArrayLike,
    std_floor: float = STD_FLOOR,
    subtract_mean: bool = True,
    name: str = "dataset",
) -> DatasetStatistics:
    """Compute :class:`DatasetStatistics` from an array of observations."""
    return DatasetStatistics.from_data(
        observations, std_floor=std_floor, subtract_mean=subtract_mean, name=name
    )


def compute_mean_std(
    observations: ArrayLike, std_floor: float = STD_FLOOR
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, std)`` per dimension with the std floored."""
    data = _as_2d_float(observations)
    return data.mean(axis=0), _safe_std(data.std(axis=0), std_floor)


def compute_std(observations: ArrayLike, std_floor: float = STD_FLOOR) -> np.ndarray:
    """Return the per-dimension standard deviation (floored) only."""
    return _safe_std(_as_2d_float(observations).std(axis=0), std_floor)


def dataset_statistics_from_array(
    observations: ArrayLike, name: str = "dataset", **kwargs: Any
) -> DatasetStatistics:
    """Alias of :func:`compute_statistics` kept for readability at call sites."""
    return compute_statistics(observations, name=name, **kwargs)
