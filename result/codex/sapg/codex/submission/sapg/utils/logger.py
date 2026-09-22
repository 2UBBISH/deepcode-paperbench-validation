"""CSV + TensorBoard logger (no heavy dependencies are required)."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict


class Logger:
    def __init__(self, logdir: str, use_tensorboard: bool = True, verbose: bool = True) -> None:
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self.verbose = verbose
        self.csv_path = os.path.join(logdir, "progress.csv")
        self._fieldnames = None
        self._writer = None
        self._handle = None
        self._tb = None
        if use_tensorboard:
            try:  # pragma: no cover - optional dependency
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=logdir)
            except Exception:
                self._tb = None

    def log(self, step: int, values: Dict[str, Any], prefix: str = "") -> None:
        row = {"step": step}
        for key, value in values.items():
            name = f"{prefix}{key}"
            row[name] = value
            if self._tb is not None:
                self._tb.add_scalar(name, value, step)
        self._write_row(row)
        if self.verbose:
            pretty = "  ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
            )
            print(f"[log {step}] {pretty}", flush=True)

    def _write_row(self, row: Dict[str, Any]) -> None:
        if self._handle is None:
            self._fieldnames = list(row.keys())
            self._handle = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._handle, fieldnames=self._fieldnames, extrasaction="ignore")
            self._writer.writeheader()
        # extend schema if new keys show up later (keeps the CSV append-friendly)
        missing = [k for k in row if k not in self._fieldnames]
        if missing:
            self._fieldnames.extend(missing)
            self._handle.close()
            rows = list(csv.DictReader(open(self.csv_path)))
            self._handle = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._handle, fieldnames=self._fieldnames, extrasaction="ignore")
            self._writer.writeheader()
            for old in rows:
                self._writer.writerow(old)
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._tb is not None:
            self._tb.close()
