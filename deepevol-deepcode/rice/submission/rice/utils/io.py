"""Filesystem / configuration helpers shared by the RICE pipeline.

Responsibilities
----------------
* load the per-environment YAML configs from ``configs/`` (``get_config``),
  including the ``default.yaml`` merging logic used by every experiment;
* persist the artefacts produced by the two stages -- mask-network
  checkpoints, critical-state buffers, per-trajectory fidelity scores,
  refinement logs (``save_json`` / ``load_json``);
* guarantee that the output directories requested by a run exist
  (``ensure_dir``).
"""

from __future__ import annotations

import json
import os
import pickle
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "ensure_dir",
    "get_config",
    "merge_configs",
    "save_json",
    "load_json",
    "save_pickle",
    "load_pickle",
    "save_numpy",
    "load_numpy",
    "config_to_dict",
]

# ``rice/utils/io.py`` -> ``rice/utils`` -> ``rice`` -> project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")


def ensure_dir(path: str) -> str:
    """Create ``path`` (recursively) if needed and return it."""
    if path:
        os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# JSON serialisation that tolerates numpy scalars / arrays / torch tensors
# ---------------------------------------------------------------------------
def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "detach"):  # torch tensor
        return obj.detach().cpu().numpy().tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(obj: Any, path: str, indent: int = 2) -> str:
    """Save ``obj`` as JSON, creating parent directories on demand."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
    return path


def load_json(path: str, default: Optional[Any] = None) -> Any:
    """Load a JSON file; return ``default`` when it does not exist."""
    if not os.path.exists(path):
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with open(path, "r") as fh:
        return json.load(fh)


def save_pickle(obj: Any, path: str) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    return path


def load_pickle(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_numpy(array: Any, path: str) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    np.save(path, np.asarray(array))
    return path


def load_numpy(path: str):
    return np.load(path, allow_pickle=True)


# ---------------------------------------------------------------------------
# Configuration handling
# ---------------------------------------------------------------------------
def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge (``override`` wins); returns a new dict."""
    out = deepcopy(base or {})
    for key, value in (override or {}).items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = merge_configs(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _read_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"config not found: {path}")
    if yaml is None:  # pragma: no cover - fallback for minimal installs
        raise ImportError("pyyaml is required to read configs")
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def get_config(name: str = "default", config_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load a YAML config, merged over ``default.yaml``.

    ``name`` may be a config stem (``"hopper"``), a file name
    (``"hopper.yaml"``) or an absolute path.  When the requested config is
    ``default`` the file is returned as-is.
    """
    config_dir = config_dir or CONFIG_DIR

    if os.path.isabs(name) or os.sep in name:
        path = name if name.endswith((".yaml", ".yml")) else name + ".yaml"
        cfg = _read_yaml(path)
        base = merge_configs(_read_yaml(os.path.join(config_dir, "default.yaml")) if
                             os.path.exists(os.path.join(config_dir, "default.yaml")) else {},
                             cfg)
        return base

    stem = name[:-5] if name.endswith(".yaml") else name
    path = os.path.join(config_dir, stem + ".yaml")
    cfg = _read_yaml(path)

    default_path = os.path.join(config_dir, "default.yaml")
    if stem != "default" and os.path.exists(default_path):
        cfg = merge_configs(_read_yaml(default_path), cfg)
    return cfg


def config_to_dict(config: Any) -> Dict[str, Any]:
    """Best-effort conversion of a (possibly OmegaConf) config to a plain dict."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return {k: config_to_dict(v) for k, v in config.items()}
    if hasattr(config, "items"):
        try:
            return {k: config_to_dict(v) for k, v in config.items()}
        except Exception:
            pass
    if isinstance(config, (list, tuple)):
        return [config_to_dict(v) for v in config]
    return config
