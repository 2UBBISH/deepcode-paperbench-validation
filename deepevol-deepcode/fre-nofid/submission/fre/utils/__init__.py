"""Utilities for the FRE reproduction.

This package groups dependency-light helpers used across the codebase:

- :mod:`fre.utils.logging`       -- loggers, metric tracking, seeding, JSON/CSV IO.
- :mod:`fre.utils.normalization` -- standardisation, running mean/std, return scaling.
- :mod:`fre.utils.discretize`    -- reward / XY / state discretisation arithmetic.

The submodules deliberately avoid heavy import-time dependencies (``torch`` is
optional and imported lazily) so that data preprocessing and analysis scripts can
run on CPU-only machines.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
from .logging import (  # noqa: F401
    CSVLogger,
    JsonlLogger,
    Logger,
    MetricTracker,
    RunningMeanStd,
    configure_logging,
    format_time,
    get_logger,
    progress,
    read_json,
    seed_everything,
    set_seed,
    write_json,
)

# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------
from .normalization import (  # noqa: F401
    DEFAULT_CLIP,
    EPS,
    NormalizationStats,
    Standardizer,
    clip_normalized,
    compute_mean_std,
    denormalize,
    load_stats,
    min_max_scale,
    normalize,
    normalize_returns,
    normalize_states,
    running_stats_from_batches,
    save_stats,
    stats_dict,
    to_tensor_like,
    unstandardize,
    standardize,
)

# ---------------------------------------------------------------------------
# discretize
# ---------------------------------------------------------------------------
from .discretize import (  # noqa: F401
    DEFAULT_XY_BINS,
    DEFAULT_XY_EXTENT,
    NUM_REWARD_BINS,
    REWARD_EMB_DIM,
    REWARD_MAX,
    REWARD_MIN,
    REWARD_RESCALE,
    STATE_EMB_DIM,
    TOKEN_DIM,
    bin_distance,
    bin_index_to_center,
    bin_to_xy,
    bins_to_centers,
    discretize_reward,
    discretize_state,
    discretize_value,
    discretize_xy,
    grid_size,
    is_torch_tensor,
    normalized_reward,
    one_hot,
    one_hot_reward,
    rescale_reward,
    reward_discretization_info,
    to_numpy,
    to_torch_like,
    undiscretize_reward,
    xy_distance,
)

__all__ = [
    # logging
    "CSVLogger",
    "JsonlLogger",
    "Logger",
    "MetricTracker",
    "RunningMeanStd",
    "configure_logging",
    "format_time",
    "get_logger",
    "progress",
    "read_json",
    "seed_everything",
    "set_seed",
    "write_json",
    # normalization
    "DEFAULT_CLIP",
    "EPS",
    "NormalizationStats",
    "Standardizer",
    "clip_normalized",
    "compute_mean_std",
    "denormalize",
    "load_stats",
    "min_max_scale",
    "normalize",
    "normalize_returns",
    "normalize_states",
    "running_stats_from_batches",
    "save_stats",
    "stats_dict",
    "to_tensor_like",
    "unstandardize",
    "standardize",
    # discretize
    "DEFAULT_XY_BINS",
    "DEFAULT_XY_EXTENT",
    "NUM_REWARD_BINS",
    "REWARD_EMB_DIM",
    "REWARD_MAX",
    "REWARD_MIN",
    "REWARD_RESCALE",
    "STATE_EMB_DIM",
    "TOKEN_DIM",
    "bin_distance",
    "bin_index_to_center",
    "bin_to_xy",
    "bins_to_centers",
    "discretize_reward",
    "discretize_state",
    "discretize_value",
    "discretize_xy",
    "grid_size",
    "is_torch_tensor",
    "normalized_reward",
    "one_hot",
    "one_hot_reward",
    "rescale_reward",
    "reward_discretization_info",
    "to_numpy",
    "to_torch_like",
    "undiscretize_reward",
    "xy_distance",
]
