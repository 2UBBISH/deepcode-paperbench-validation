"""Utility sub-package for the SAPG reproduction.

Exposes the shared helpers used across the code base:

* :mod:`sapg.utils.config`   -- central hyper-parameter configuration (Appendix B
  Tables 2-4) via :class:`SAPGConfig` and :func:`build_config`.
* :mod:`sapg.utils.gae`      -- Generalized Advantage Estimation (paper ``tau``
  interpreted as the GAE lambda).
* :mod:`sapg.utils.returns`  -- n-step on-policy and 1-step off-policy critic
  targets (Section 4.1) plus discounted-return helpers.
* :mod:`sapg.utils.logging`  -- lightweight tensorboard/console logging.

All heavy imports (torch, PyYAML) are deferred with PEP 562 module-level
``__getattr__`` so ``import sapg.utils`` stays cheap and side-effect free.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

__all__: List[str] = [
    # --- config -----------------------------------------------------------
    "SAPGConfig",
    "build_config",
    "ensure_dir",
    "TOTAL_ENVS",
    "NUM_POLICIES",
    "TOTAL_TRANSITIONS",
    "TASK_DEFAULTS",
    "TASK_OVERRIDES",
    # --- returns / targets ------------------------------------------------
    "compute_n_step_targets",
    "compute_one_step_targets",
    "compute_on_policy_targets",
    "compute_off_policy_targets",
    "compute_value_targets",
    "compute_returns",
    "compute_discounted_returns",
    "compute_advantages_and_returns",
    "normalize_advantages",
    "DEFAULT_GAMMA",
    "DEFAULT_LAMBDA",
    "DEFAULT_N_STEP",
    # --- GAE --------------------------------------------------------------
    "compute_gae",
    "compute_gae_for_buffer",
    "GeneralizedAdvantageEstimator",
    "make_gae",
    "GAEStats",
    "DEFAULT_TAU",
    "DEFAULT_GAE_LAMBDA",
    # --- logging ----------------------------------------------------------
    "Logger",
    "ExperimentLogger",
    "MetricLogger",
    "make_logger",
    "get_logger",
    "AverageMeter",
    "RunningMeanStd",
]


#: Mapping of public symbol -> owning submodule (single source of truth used by
#: ``__getattr__`` and kept in sync with ``__all__`` above).
_LAZY_EXPORTS: Dict[str, str] = {}


def _register(names: List[str], module: str) -> None:
    for _name in names:
        _LAZY_EXPORTS[_name] = module


_register(
    [
        "SAPGConfig",
        "build_config",
        "ensure_dir",
        "TOTAL_ENVS",
        "NUM_POLICIES",
        "TOTAL_TRANSITIONS",
        "TASK_DEFAULTS",
        "TASK_OVERRIDES",
    ],
    "config",
)

_register(
    [
        "compute_n_step_targets",
        "compute_one_step_targets",
        "compute_on_policy_targets",
        "compute_off_policy_targets",
        "compute_value_targets",
        "compute_returns",
        "compute_discounted_returns",
        "compute_advantages_and_returns",
        "normalize_advantages",
        "DEFAULT_GAMMA",
        "DEFAULT_LAMBDA",
        "DEFAULT_N_STEP",
    ],
    "returns",
)

_register(
    [
        "compute_gae",
        "compute_gae_for_buffer",
        "GeneralizedAdvantageEstimator",
        "make_gae",
        "GAEStats",
        "DEFAULT_TAU",
        "DEFAULT_GAE_LAMBDA",
    ],
    "gae",
)

_register(
    [
        "Logger",
        "ExperimentLogger",
        "MetricLogger",
        "make_logger",
        "get_logger",
        "AverageMeter",
        "RunningMeanStd",
    ],
    "logging",
)


#: Module-level caches imported eagerly because they are pure-python helpers
#: used by nearly every other module.  Failures here must never break imports.
def _config_module():
    return importlib.import_module(".config", __name__)


def _gaussian_angle_reference() -> float:
    """Return the normal-approximation z-value used by the paper's shaded band.

    Section 5.2 reports ``mean +/- (2/sqrt(n)) * sum_i (y(t) - y_i(t))^2``; the
    constant is kept here so scripts and the logging helpers share one source.
    """

    return 2.0


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute resolution for the utility sub-package."""

    if name in _LAZY_EXPORTS:
        module_name = _LAZY_EXPORTS[name]
        try:
            module = importlib.import_module(f".{module_name}", __name__)
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise AttributeError(
                f"module 'sapg.utils' cannot resolve {name!r} "
                f"(failed to import 'sapg.utils.{module_name}': {exc})"
            ) from exc
        try:
            value = getattr(module, name)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                f"module 'sapg.utils.{module_name}' has no attribute {name!r}"
            ) from exc
        globals()[name] = value
        return value

    if name in ("band_scale", "PAPER_BAND_SCALE"):
        return _gaussian_angle_reference()

    raise AttributeError(f"module 'sapg.utils' has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(__all__))


def available_modules() -> List[str]:
    """Return the sorted list of sub-modules this package wraps."""

    return sorted(set(_LAZY_EXPORTS.values()))


def load_config(path: str) -> Any:
    """Load a YAML configuration file into a :class:`SAPGConfig`.

    Thin convenience wrapper so experimental scripts can do
    ``from sapg.utils import load_config`` without importing ``config``
    explicitly.
    """

    config_mod = _config_module()
    return config_mod.SAPGConfig.from_yaml(path)


def save_config(config: Any, path: str) -> str:
    """Persist a configuration object to ``path`` (YAML) and return the path."""

    config_mod = _config_module()
    if hasattr(config, "save_yaml"):
        config.save_yaml(path)
    else:
        import os

        import yaml

        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as handle:
            yaml.safe_dump(dict(config), handle, sort_keys=False)
    return path
