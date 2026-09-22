"""Configuration utilities: YAML loading with dotted-key access and sane defaults."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml


class Config(dict):
    """A dict with attribute-style access and recursive merging."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, item: str) -> None:
        try:
            del self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc


def _to_config(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def load_config(*paths: str) -> Config:
    """Load and merge one or more YAML config files (later files override earlier)."""
    merged: Dict[str, Any] = {}
    for path in paths:
        if not path:
            continue
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        merged = deep_merge(merged, data)
    return _to_config(merged)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def save_config(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def config_to_dict(cfg: Any) -> Any:
    if isinstance(cfg, dict):
        return {k: config_to_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [config_to_dict(v) for v in cfg]
    return cfg


# ---------------------------------------------------------------------------
# Default FOA hyper-parameters (paper main text + Appendix B.2)
# ---------------------------------------------------------------------------
FOA_DEFAULTS: Dict[str, Any] = {
    # model
    "model": {
        "name": "vit_base_patch16_224",
        "checkpoint": None,  # local npz/ckpt; None -> timm default weights
        "precision": 32,  # 32 / 8 / 6
        "num_classes": 1000,
    },
    # prompt
    "prompt": {
        "num_prompts": 3,  # N_p  (Section 4 Implementation Details)
        "init": "uniform",  # uniform initialization
        "init_range": 0.01,  # U(-init_range, init_range) -- documented default
    },
    # CMA-ES
    "cma": {
        "population_size": 28,  # K = 4 + 3*log(prompt_dim) (Hansen 2016)
        "sigma0": 1.0,  # tau^(0) = 1
        "mean0": 0.0,  # m^(0) = 0
        "cov0": 1.0,  # Sigma^(0) = I
        "seed": 0,
    },
    # fitness (Eqn. 5)
    "fitness": {
        "lambda": 0.4,  # scaled by BS/64 in the runner
        "lambda_base": 0.4,
        "beta": 1.0,  # EMA factor for test statistics in Eqn. (5); 1.0 = batch stats
        "use_entropy": True,
        "use_discrepancy": True,
        "eps": 1e-6,
    },
    # activation shifting (Eqn. 7-9)
    "shifting": {
        "enabled": True,
        "gamma": 1.0,  # step size in Eqn. (7)
        "alpha": 0.1,  # EMA factor in Eqn. (9)
    },
    # data / evaluation
    "data": {
        "dataset": "imagenet-c",
        "corruption": None,  # None -> all 15
        "severity": 5,
        "root": "./data",
        "batch_size": 64,
        "num_workers": 4,
        "shuffle": False,  # online, order-preserving stream
        "subset": "matched-frequency",  # ImageNet-V2 subset
    },
    # source statistics
    "source_stats": {
        "path": "./checkpoints/source_stats_vit_base.pt",
        "num_samples": 32,  # Q (Section 3.1 / Figure 2c)
        "seed": 0,
        "path_id": "./data",
    },
    "eval": {"ece_bins": 15},
    "seed": 0,
    "output_dir": "./outputs",
}
