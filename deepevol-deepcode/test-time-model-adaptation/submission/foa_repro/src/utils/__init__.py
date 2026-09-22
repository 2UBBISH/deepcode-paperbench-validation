"""Utilities for the FOA (Forward-Optimization Adaptation) reproduction.

This sub-package contains *glue only*: YAML config loading, deterministic
seeding, structured logging / result serialisation and checkpoint I/O.  No
paper-specified math lives here.

Modules
-------
config          : nested attribute-accessible configuration + ``FOA_DEFAULTS``.
seeding         : single global seed for python/numpy/torch + CMA-ES stream.
logging_utils   : logger setup, config pretty-printing, ASCII tables, JSON I/O.
checkpoint_io   : torch checkpoint helpers for source stats / prompt / CMA state.
"""

from __future__ import annotations

__all__ = ["config", "seeding", "logging_utils", "checkpoint_io"]

__version__ = "0.1.0"
