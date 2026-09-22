"""Lightweight logging + metric bookkeeping used by the runners.

The runners are long-lived (hundreds of millions of environment steps) and are
usually executed on remote machines, therefore we keep the dependency surface
small: stdout logging plus optional TensorBoard writers.
"""
from __future__ import annotations

import csv
import logging
import os
import sys
from typing import Any, Dict, Optional

_LOGGERS: Dict[str, logging.Logger] = {}


def get_logger(name: str = "ftrl", level: int = logging.INFO) -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    _LOGGERS[name] = logger
    return logger


class MetricLogger:
    """Accumulates scalar metrics and flushes them to TensorBoard and/or CSV."""

    def __init__(self, log_dir: str, use_tensorboard: bool = True) -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._writer = None
        self._buffer: Dict[str, Any] = {}
        self._csv_path = os.path.join(log_dir, "metrics.csv")
        self._csv_fields = None
        if use_tensorboard:
            try:  # pragma: no cover - optional dependency
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=log_dir)
            except Exception:
                self._writer = None

    # ------------------------------------------------------------------ #
    def add(self, key: str, value: float) -> None:
        self._buffer[key] = value

    def add_dict(self, values: Dict[str, Any]) -> None:
        for key, value in values.items():
            self.add(key, value)

    def flush(self, step: int) -> None:
        if self._writer is not None:  # pragma: no cover - requires tensorboard
            for key, value in self._buffer.items():
                try:
                    self._writer.add_scalar(key, float(value), step)
                except (TypeError, ValueError):
                    continue
            self._writer.flush()

        if self._buffer:
            write_header = not os.path.exists(self._csv_path)
            with open(self._csv_path, "a", newline="", encoding="utf-8") as handle:
                fields = ["step"] + sorted(self._buffer.keys())
                writer = csv.DictWriter(handle, fieldnames=fields)
                if write_header:
                    writer.writeheader()
                row = {"step": step}
                row.update({k: self._buffer[k] for k in fields if k != "step"})
                writer.writerow(row)

    def close(self) -> None:
        if self._writer is not None:  # pragma: no cover
            self._writer.close()


def human_format(number: float) -> str:
    """Compact representation used in progress prints (e.g. 12.5M)."""
    for unit in ["", "k", "M", "B", "T"]:
        if abs(number) < 1000.0:
            return f"{number:.3g}{unit}"
        number /= 1000.0
    return f"{number:.3g}P"


def log_config(cfg: Any, logger: Optional[logging.Logger] = None) -> None:
    logger = logger or get_logger()
    dump = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    for key in sorted(dump):
        logger.info("config.%s = %s", key, dump[key])
