"""Small YAML config helper (dot-access + deep merge + CLI overrides)."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml


class Config(dict):
    """``dict`` with attribute access and nested wrapping."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            self[key] = self._wrap(value)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, Config):
            return value
        if isinstance(value, Mapping):
            return Config(value)
        if isinstance(value, (list, tuple)):
            return type(value)(Config._wrap(v) for v in value)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, self._wrap(value))

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    # ------------------------------------------------------------------ #
    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], Mapping):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        def unwrap(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {k: unwrap(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [unwrap(v) for v in value]
            return value

        return unwrap(dict(self))


def merge_configs(base: Mapping[str, Any], override: Mapping[str, Any]) -> Config:
    """Recursively merge ``override`` into a copy of ``base``."""
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge_configs(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return Config(out)


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Config:
    """Load a YAML file (resolving ``defaults`` recursively) with overrides."""
    path = os.path.abspath(path)
    cfg = _load_with_defaults(path)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is not of the form key=value")
        key, raw = item.split("=", 1)
        cfg.set_path(key.strip(), yaml.safe_load(raw))

    cfg.setdefault("config_path", path)
    return cfg


def _load_with_defaults(path: str) -> Config:
    with open(path, "r") as handle:
        cfg = Config(yaml.safe_load(handle) or {})
    # A config may declare a `defaults` list whose entries are merged first so
    # that task configs stay short (mirrors the DexPBT/IsaacGym layout).
    defaults = cfg.pop("defaults", None)
    if not defaults:
        return cfg
    base = Config()
    for name in defaults:
        base = merge_configs(base, _load_with_defaults(_resolve(path, name)))
    return merge_configs(base, cfg)


def _resolve(config_path: str, name: str) -> str:
    candidate = os.path.join(os.path.dirname(os.path.abspath(config_path)), name)
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(name):
        return name
    raise FileNotFoundError(f"Could not resolve default config '{name}'")
