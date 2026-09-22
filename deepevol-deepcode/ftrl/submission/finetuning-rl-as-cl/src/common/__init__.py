"""Shared scaffolding utilities for the fine-tuning-as-continual-learning codebase.

This sub-package contains environment-agnostic helpers that are reused by every
training / evaluation entry point:

* :mod:`src.common.config`       -- YAML config loading, attribute access, CLI overrides.
* :mod:`src.common.seeding`      -- deterministic seeding of python/numpy/torch/gym.
* :mod:`src.common.logging_utils`-- logger factory, TensorBoard/CSV metric logging.
* :mod:`src.common.checkpointing`-- step-tagged checkpoint save / load / resume helpers.

Nothing in this module depends on an environment-specific package (NetHack,
Montezuma, Meta-World), so it can be imported in CPU-only environments.
"""

from __future__ import annotations

from .checkpointing import (
    ensure_dir,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    maybe_resume,
    save_checkpoint,
)
from .config import Config, apply_overrides, dump_config, load_config
from .logging_utils import MetricLogger, get_logger, human_format, log_config
from .seeding import make_generator, seed_env, set_seed

__all__ = [
    # config
    "Config",
    "load_config",
    "apply_overrides",
    "dump_config",
    # seeding
    "set_seed",
    "make_generator",
    "seed_env",
    # logging
    "get_logger",
    "MetricLogger",
    "human_format",
    "log_config",
    # checkpointing
    "ensure_dir",
    "save_checkpoint",
    "load_checkpoint",
    "list_checkpoints",
    "latest_checkpoint",
    "maybe_resume",
]
