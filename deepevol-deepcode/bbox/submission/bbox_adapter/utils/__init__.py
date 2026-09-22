"""Utility package for BBox-Adapter.

This package bundles the small, dependency-light helpers that every experiment
entry point needs:

* :mod:`bbox_adapter.utils.seed` -- deterministic seeding for python / numpy /
  torch plus per-component sub-seed derivation (``set_seed``,
  ``seed_everything``, ``derive_component_seeds``, ...).
* :mod:`bbox_adapter.utils.logging` -- console/file logging setup,
  ``RunLogger`` run-directory management, ``MetricTracker`` bookkeeping,
  JSON/JSONL/CSV artifact writing and checkpoint naming.

It additionally exposes a tiny YAML configuration loader (``load_config`` /
``deep_merge``) used by the ``scripts/run_*.py`` drivers to combine
``configs/default.yaml`` with the per-dataset YAML overrides.

Nothing in this package ever touches the black-box LLM: utilities only handle
seeds, logs, artifacts and plain configuration mappings.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Seeding utilities (re-exported from .seed)
# ---------------------------------------------------------------------------
from .seed import (  # noqa: F401
    COMPONENT_NAMES,
    DEFAULT_SEED,
    SeedContext,
    capture_state,
    derive_component_seeds,
    describe_seed_state,
    numpy_rng,
    restore_state,
    rng_from_seed,
    seed_everything,
    seed_worker,
    set_seed,
    spawn_seeds,
    temporary_seed,
    torch_generator,
    torch_initial_seed,
)

# ---------------------------------------------------------------------------
# Logging / artifact utilities (re-exported from .logging)
#
# The import is guarded so that a partially-available environment can still use
# the seeding helpers (and vice versa).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - defensive
    from .logging import (  # noqa: F401
        DATE_FORMAT,
        DEFAULT_LOG_LEVEL,
        LEVELS,
        LOG_FORMAT,
        ColoredFormatter,
        MetricTracker,
        RunLogger,
        Timer,
        append_metrics,
        checkpoint_name,
        checkpoint_path,
        count_checkpoints,
        describe_environment,
        format_seconds,
        get_logger,
        json_safe,
        kv_table,
        latest_checkpoint,
        list_checkpoints,
        load_metrics,
        log_config,
        make_run_dir,
        mean_of,
        progress,
        read_curve_csv,
        read_jsonl,
        resolve_checkpoint,
        setup_logging,
        summary_stats,
        table,
        to_jsonable,
        write_curve_csv,
        write_csv,
        write_json,
        write_jsonl,
    )

    _LOGGING_AVAILABLE = True
    _LOGGING_EXPORTS = [
        "DATE_FORMAT",
        "DEFAULT_LOG_LEVEL",
        "LEVELS",
        "LOG_FORMAT",
        "ColoredFormatter",
        "MetricTracker",
        "RunLogger",
        "Timer",
        "append_metrics",
        "checkpoint_name",
        "checkpoint_path",
        "count_checkpoints",
        "describe_environment",
        "format_seconds",
        "get_logger",
        "json_safe",
        "kv_table",
        "latest_checkpoint",
        "list_checkpoints",
        "load_metrics",
        "log_config",
        "make_run_dir",
        "mean_of",
        "progress",
        "read_curve_csv",
        "read_jsonl",
        "resolve_checkpoint",
        "setup_logging",
        "summary_stats",
        "table",
        "to_jsonable",
        "write_curve_csv",
        "write_csv",
        "write_json",
        "write_jsonl",
    ]
except Exception as _logging_exc:  # pragma: no cover - defensive
    _LOGGING_AVAILABLE = False
    _LOGGING_EXPORTS = []
    _LOGGING_IMPORT_ERROR = _logging_exc


# ---------------------------------------------------------------------------
# Configuration loading (paper-silent convenience default)
# ---------------------------------------------------------------------------
def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` on top of ``base``.

    Neither input is mutated; nested mappings are merged key-by-key and any
    non-mapping value in ``override`` replaces the base value outright. This is
    what the per-dataset configs rely on: ``configs/strategyqa.yaml`` overrides
    ``configs/default.yaml`` section by section.

    Args:
        base: base configuration mapping.
        override: mapping whose values win.

    Returns:
        A new merged dictionary.
    """
    if not isinstance(base, Mapping):
        return copy.deepcopy(dict(override)) if not isinstance(override, Mapping) else copy.deepcopy(override)
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    if not isinstance(override, Mapping):
        return merged
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def read_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file into a plain ``dict`` (requires ``pyyaml``).

    Raises:
        ImportError: if ``yaml`` is not installed.
        FileNotFoundError: if ``path`` does not exist.
    """
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Loading YAML configuration requires pyyaml (`pip install pyyaml`)."
        ) from exc
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"config file {path} did not parse to a mapping")
    return dict(data)


def write_yaml(path: str, payload: Mapping[str, Any]) -> str:
    """Serialize ``payload`` to YAML at ``path`` (requires ``pyyaml``)."""
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Writing YAML configuration requires pyyaml (`pip install pyyaml`)."
        ) from exc
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False, default_flow_style=False)
    return path


def resolve_config_path(name_or_path: str) -> str:
    """Resolve a dataset name (``"gsm8k"``) or a config path to a YAML file.

    Resolution order:
      1. the literal path, if it exists;
      2. ``<name>.yaml`` in the current directory;
      3. ``<name>.yaml`` in the bundled ``bbox_adapter/configs`` directory.

    Raises:
        FileNotFoundError: if nothing matches.
    """
    candidate = str(name_or_path)
    if candidate.endswith((".yaml", ".yml")) and os.path.exists(candidate):
        return os.path.abspath(candidate)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    stem = os.path.splitext(os.path.basename(candidate))[0]
    local = f"{candidate}.yaml"
    if os.path.exists(local):
        return os.path.abspath(local)
    configs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    for fname in (f"{stem}.yaml", f"{stem}.yml", candidate):
        path = os.path.join(configs_dir, fname)
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        f"could not resolve configuration {name_or_path!r} "
        f"(looked in {os.getcwd()} and {configs_dir})"
    )


def default_config_path() -> str:
    """Absolute path of the bundled ``configs/default.yaml``."""
    configs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    return os.path.join(configs_dir, "default.yaml")


def config_to_dataclass_kwargs(config: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    """Pick ``keys`` present in ``config`` (helper for dataclass construction)."""
    out: Dict[str, Any] = {}
    for key in keys:
        if key in config:
            out[key] = config[key]
    return out


def load_config(
    path: Optional[Any] = None,
    *,
    base: Optional[Any] = None,
    dataset: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    with_defaults: bool = True,
) -> Dict[str, Any]:
    """Load a BBox-Adapter configuration.

    Mirrors the plan's config strategy: the bundled ``configs/default.yaml`` is
    loaded first (unless ``with_defaults=False``) and the requested per-dataset
    file is deep-merged on top of it.

    Args:
        path: config path, dataset name, or an already-loaded mapping. When
            ``None`` and ``dataset`` is given, the dataset name is used.
        base: explicit base config (path/mapping); defaults to
            ``configs/default.yaml`` when available.
        dataset: dataset name used when ``path`` is ``None``.
        overrides: extra values merged last (e.g. CLI flags).
        with_defaults: whether to start from ``configs/default.yaml``.

    Returns:
        The merged configuration dictionary (contains a ``"config_path"`` key
        recording the file actually loaded, and ``"config_base"`` for the base).
    """
    if isinstance(path, Mapping):
        return deep_merge(dict(path), overrides or {})

    base_cfg: Dict[str, Any] = {}
    base_used: Optional[str] = None
    if base is None and with_defaults:
        candidate = default_config_path()
        if os.path.exists(candidate):
            base = candidate
    if isinstance(base, Mapping):
        base_cfg = dict(base)
    elif isinstance(base, str):
        base_cfg = read_yaml(resolve_config_path(base))
        base_used = resolve_config_path(base)

    target = path if path is not None else dataset
    cfg: Dict[str, Any] = dict(base_cfg)
    loaded_path: Optional[str] = None
    if target is not None:
        loaded_path = resolve_config_path(str(target))
        override_cfg = read_yaml(loaded_path)
        # Avoid re-applying default.yaml when the caller asked for it directly.
        if base_used is not None and os.path.abspath(loaded_path) == os.path.abspath(base_used):
            cfg = dict(base_cfg)
        else:
            cfg = deep_merge(base_cfg, override_cfg)
        if "dataset" not in cfg and dataset:
            cfg["dataset"] = dataset
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg.setdefault("config_path", loaded_path)
    cfg.setdefault("config_base", base_used)
    return cfg


def dumps_config(config: Mapping[str, Any]) -> str:
    """JSON dump with the JSON-safe conversion from :mod:`logging` applied."""
    try:
        return json.dumps(json_safe(config), indent=2)  # type: ignore[name-defined]
    except Exception:  # pragma: no cover - defensive
        return json.dumps(dict(config), default=str, indent=2)


__all__: List[str] = [
    # seeding
    "COMPONENT_NAMES",
    "DEFAULT_SEED",
    "SeedContext",
    "capture_state",
    "derive_component_seeds",
    "describe_seed_state",
    "numpy_rng",
    "restore_state",
    "rng_from_seed",
    "seed_everything",
    "seed_worker",
    "set_seed",
    "spawn_seeds",
    "temporary_seed",
    "torch_generator",
    "torch_initial_seed",
    # config loading (paper-silent convenience defaults)
    "deep_merge",
    "read_yaml",
    "write_yaml",
    "load_config",
    "resolve_config_path",
    "default_config_path",
    "config_to_dataclass_kwargs",
    "dumps_config",
] + _LOGGING_EXPORTS


def has_logging() -> bool:
    """Whether :mod:`bbox_adapter.utils.logging` imported successfully."""
    return bool(_LOGGING_AVAILABLE)


def _self_test() -> Dict[str, Any]:
    """Dependency-free smoke test (``python -m bbox_adapter.utils``)."""
    result: Dict[str, Any] = {}

    # Seeding
    effective = seed_everything(DEFAULT_SEED)
    assert effective == DEFAULT_SEED
    seeds = derive_component_seeds(DEFAULT_SEED)
    assert "adapter_init" in seeds and "blackbox_sampling" in seeds
    assert len(spawn_seeds(0, 3)) == 3

    # Config deep-merge semantics
    merged = deep_merge(
        {"training": {"lr": 5e-6, "batch_size": 64}, "dataset": None},
        {"training": {"lr": 1e-5}, "dataset": "gsm8k"},
    )
    assert merged["training"]["lr"] == 1e-5
    assert merged["training"]["batch_size"] == 64
    assert merged["dataset"] == "gsm8k"
    # base must not be mutated
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"b": 2}})
    assert base["a"]["b"] == 1

    result["logging_available"] = _LOGGING_AVAILABLE
    try:
        path = resolve_config_path("strategyqa")
        cfg = load_config(path)
        result["config_keys"] = sorted(k for k in cfg.keys() if not k.startswith("config_"))
    except FileNotFoundError:
        result["config_keys"] = []
    return result


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(_self_test(), indent=2))
