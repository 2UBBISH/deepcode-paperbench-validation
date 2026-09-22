"""Configuration loading utilities.

Configs live in ``configs/*.yaml``.  This module turns a YAML file into a
nested, attribute-accessible ``Config`` object and supports command line
overrides of the form ``--set a.b=c``.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable, List, Optional

try:  # PyYAML is a hard dependency of the project but keep import forgiving.
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class Config(dict):
    """A dict with attribute access and recursive ``Config`` values."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            self[key] = self._wrap(value)

    # ------------------------------------------------------------------ #
    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, Config):
            return value
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, self._wrap(value))

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, Config):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [v.to_dict() if isinstance(v, Config) else v for v in value]
            else:
                out[key] = value
        return out

    def merge(self, other: Dict[str, Any]) -> "Config":
        for key, value in other.items():
            if isinstance(value, dict) and isinstance(self.get(key), dict):
                self[key].merge(value)
            else:
                self[key] = value
        return self

    def copy(self) -> "Config":  # type: ignore[override]
        return Config(copy.deepcopy(self.to_dict()))


def _coerce(text: str) -> Any:
    """Best-effort conversion of a CLI string into a python scalar."""
    lowered = text.strip().lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"none", "null", "~"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith("[") or text.startswith("{"):
        if yaml is not None:
            try:
                return yaml.safe_load(text)
            except Exception:  # pragma: no cover
                pass
    return text


def apply_overrides(cfg: Config, overrides: Iterable[str]) -> Config:
    """Apply ``a.b.c=value`` style overrides to ``cfg`` in place."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Malformed override '{override}', expected key=value")
        key, value = override.split("=", 1)
        node: Any = cfg
        parts: List[str] = key.strip().split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = Config({})
            node = node[part]
        node[parts[-1]] = _coerce(value)
    return cfg


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Config:
    """Load a YAML config and optionally apply overrides."""
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load configuration files")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = Config(raw)
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg


def dump_config(cfg: Config, path: str) -> None:
    """Persist a config back to disk (used to snapshot runs)."""
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required to dump configuration files")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=True)
