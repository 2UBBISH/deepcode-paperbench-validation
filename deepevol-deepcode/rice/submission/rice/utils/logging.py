"""Minimal logging helpers used by every RICE entry point.

The schedules reproduce the paper's reporting granularity:

* Stage 1 (``train_mask``) reports the mask-network training *time* for a
  fixed sample budget -- Table 4 of the paper.
* Stage 2 (``run_refine``) reports the episode return of the refining
  policy (``refine/reward``) together with the RND bonus statistics.
* Experiment I reports the per-window fidelity score.

Everything is written both to stdout and to a ``log.txt`` inside the run
directory so that the ``experiments/*`` drivers can post-process them.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Dict, List, Optional

from .io import ensure_dir, save_json

__all__ = ["get_logger", "Logger", "format_mean_std", "set_log_level"]

_LOG_FORMAT = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str = "rice", out_dir: Optional[str] = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stdout and (optionally) ``out_dir/log.txt``."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(stream)

    if out_dir:
        ensure_dir(out_dir)
        path = os.path.join(out_dir, "log.txt")
        if not any(getattr(h, "baseFilename", None) == os.path.abspath(path)
                   for h in logger.handlers):
            fh = logging.FileHandler(path)
            fh.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
            logger.addHandler(fh)
    return logger


def set_log_level(level) -> None:
    logging.getLogger("rice").setLevel(level)


def format_mean_std(values: List[float], decimals: int = 2) -> str:
    """Format ``[v1, v2, ...]`` as ``"mean +- std"`` (Table 1 / Table 4 style)."""
    import numpy as np

    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return "n/a"
    if arr.size == 1:
        return f"{arr.mean():.{decimals}f}"
    return f"{arr.mean():.{decimals}f} +- {arr.std():.{decimals}f}"


class Logger:
    """Thin experiment logger: key/value series + wall-clock timers + JSON dump.

    Typical use::

        logger = Logger(out_dir="outputs/hopper/rice", config=cfg)
        t0 = logger.timer_start("reward")
        logger.record(reward=ep_ret, step=global_step)
        logger.timer_end("reward")
        logger.dump()

    ``record`` accepts any ``**kwargs`` so that the mask trainer can log
    ``mask_loss`` / ``blinding_bonus`` while the refiners log
    ``refine/reward`` / ``rnd/mean_intrinsic``.
    """

    def __init__(self, out_dir: Optional[str] = None, name: str = "rice",
                 config: Optional[Dict] = None, verbose: bool = True):
        self.out_dir = out_dir
        self.name = name
        self.config = config or {}
        self.history: Dict[str, List[float]] = {}
        self.timers: Dict[str, float] = {}
        self._t0: Dict[str, float] = {}
        if out_dir:
            ensure_dir(out_dir)
        self.logger = get_logger(name, out_dir)
        self.verbose = verbose
        if config:
            try:
                save_json(config, os.path.join(out_dir, "config.json")) if out_dir else None
            except Exception:
                pass

    # -- series -----------------------------------------------------------
    def record(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if value is None:
                continue
            try:
                value = float(value)
            except Exception:
                continue
            self.history.setdefault(key, []).append(value)
        if self.verbose and kwargs:
            msg = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
            self.logger.info(msg)

    def log_dict(self, data: Dict, prefix: str = "") -> None:
        flat = {}
        for key, value in (data or {}).items():
            key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                flat.update({f"{key}/{k}": v for k, v in value.items()})
            else:
                flat[key] = value
        self.record(**flat)

    # -- timers -----------------------------------------------------------
    def timer_start(self, name: str) -> float:
        self._t0[name] = time.time()
        return self._t0[name]

    def timer_end(self, name: str, accumulate: bool = True) -> float:
        """Stop ``name`` and return the elapsed seconds (Table 4 reporting)."""
        start = self._t0.pop(name, None)
        if start is None:
            return 0.0
        elapsed = time.time() - start
        if accumulate:
            self.timers[name] = self.timers.get(name, 0.0) + elapsed
        else:
            self.timers[name] = elapsed
        return elapsed

    # -- persistence ------------------------------------------------------
    def dump(self, filename: str = "progress.json") -> Optional[str]:
        if not self.out_dir:
            return None
        payload = {"history": self.history, "timers": self.timers}
        for key, series in self.history.items():
            payload[f"{key}/mean"] = float(sum(series) / len(series)) if series else None
            payload[f"{key}/last"] = series[-1] if series else None
        return save_json(payload, os.path.join(self.out_dir, filename))

    def close(self) -> None:
        self.dump()
        for handler in list(self.logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)
