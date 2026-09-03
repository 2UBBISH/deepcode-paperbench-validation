"""Lightweight logging utilities for RICE experiments.

Supports CSV logging, optional TensorBoard summaries, and console progress
printing.  Designed to be backend-agnostic so that training scripts can log
metrics without depending on a specific experiment-tracking service.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class Logger:
    """Unified CSV / TensorBoard / console logger.

    Parameters
    ----------
    log_dir : str or Path, optional
        Directory where ``metrics.csv`` (and optionally TensorBoard events)
        are written.  If ``None``, only console logging is performed.
    experiment_name : str, optional
        Name used for the TensorBoard sub-directory and console prefixes.
    use_tensorboard : bool, optional
        Whether to write TensorBoard event files.  Requires ``tensorboard``.
    use_csv : bool, optional
        Whether to write a ``metrics.csv`` file.
    verbose : bool, optional
        Whether to print metrics to stdout on every ``log`` call.
    """

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        experiment_name: str = "rice",
        use_tensorboard: bool = False,
        use_csv: bool = True,
        verbose: bool = True,
    ) -> None:
        self.log_dir: Optional[Path] = Path(log_dir) if log_dir is not None else None
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        self.use_csv = use_csv
        self.verbose = verbose

        self._start_time = time.time()
        self._step = 0
        self._writer: Optional[Any] = None
        self._csv_path: Optional[Path] = None
        self._csv_file: Optional[Any] = None
        self._csv_writer: Optional[Any] = None
        self._headers: List[str] = []

        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

            if self.use_csv:
                self._csv_path = self.log_dir / "metrics.csv"
                self._csv_file = open(self._csv_path, "a", newline="")
                self._csv_writer = csv.writer(self._csv_file)

            if self.use_tensorboard:
                try:
                    from torch.utils.tensorboard import SummaryWriter

                    tb_dir = self.log_dir / "tensorboard" / self.experiment_name
                    tb_dir.mkdir(parents=True, exist_ok=True)
                    self._writer = SummaryWriter(log_dir=str(tb_dir))
                except Exception as exc:  # pragma: no cover
                    print(
                        f"[Logger] TensorBoard unavailable ({exc}); disabling.",
                        file=sys.stderr,
                    )
                    self.use_tensorboard = False

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """Log a dictionary of scalar metrics.

        Parameters
        ----------
        metrics : dict
            Mapping from metric name to scalar (or convertible) value.
        step : int, optional
            Global step index.  If ``None``, an internal counter is used.
        prefix : str, optional
            Optional prefix prepended to metric names in TensorBoard / CSV.
        """
        if step is None:
            step = self._step
            self._step += 1
        else:
            self._step = max(self._step, step + 1)

        elapsed = time.time() - self._start_time
        row: Dict[str, Any] = {"step": step, "time_elapsed": elapsed}

        for key, value in metrics.items():
            full_key = f"{prefix}/{key}" if prefix else key
            scalar = _to_scalar(value)
            row[full_key] = scalar

            if self.use_tensorboard and self._writer is not None:
                self._writer.add_scalar(full_key, scalar, global_step=step)

        if self.use_csv:
            self._write_csv_row(row)

        if self.verbose:
            items = ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in row.items() if k not in ("step", "time_elapsed"))
            print(f"[{self.experiment_name}] step={step} elapsed={elapsed:.1f}s {items}")

    def log_histogram(
        self,
        tag: str,
        values: Union[List[float], np.ndarray],
        step: Optional[int] = None,
    ) -> None:
        """Log a histogram summary (TensorBoard only)."""
        if not self.use_tensorboard or self._writer is None:
            return
        if step is None:
            step = self._step
        values = np.asarray(values, dtype=np.float32)
        self._writer.add_histogram(tag, values, global_step=step)

    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """Log a text summary (TensorBoard only)."""
        if not self.use_tensorboard or self._writer is None:
            return
        if step is None:
            step = self._step
        self._writer.add_text(tag, text, global_step=step)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Persist hyper-parameters to a ``hparams.yaml`` file and TensorBoard."""
        if self.log_dir is None:
            return
        import yaml

        hparams_path = self.log_dir / "hparams.yaml"
        with open(hparams_path, "w") as f:
            yaml.safe_dump(params, f, default_flow_style=False)

        if self.use_tensorboard and self._writer is not None:
            try:
                self._writer.add_hparams(params, {})
            except Exception:
                pass

    def close(self) -> None:
        """Flush and close open files / writers."""
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _write_csv_row(self, row: Dict[str, Any]) -> None:
        if self._csv_writer is None:
            return

        headers = ["step", "time_elapsed"] + sorted(
            [k for k in row.keys() if k not in ("step", "time_elapsed")]
        )

        if headers != self._headers:
            self._headers = headers
            self._csv_file.seek(0)
            self._csv_file.truncate()
            self._csv_writer.writerow(self._headers)

        self._csv_writer.writerow([row.get(h, "") for h in self._headers])
        self._csv_file.flush()

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class ConsoleLogger:
    """Minimal console-only logger for quick scripts and tests."""

    def __init__(self, name: str = "rice", verbose: bool = True) -> None:
        self.name = name
        self.verbose = verbose
        self._start = time.time()

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = "") -> None:
        if not self.verbose:
            return
        elapsed = time.time() - self._start
        step_str = f"step={step}" if step is not None else ""
        items = ", ".join(
            f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        print(f"[{self.name}] {step_str} elapsed={elapsed:.1f}s {items}")

    def close(self) -> None:
        pass


def _to_scalar(value: Any) -> Union[int, float]:
    """Convert a metric value to a Python scalar."""
    if isinstance(value, (int, float)):
        return value
    if hasattr(value, "item"):
        return value.item()
    arr = np.asarray(value)
    if arr.size == 1:
        return arr.item()
    raise ValueError(f"Cannot convert value of shape {arr.shape} to scalar")


def make_logger(
    log_dir: Optional[Union[str, Path]] = None,
    experiment_name: str = "rice",
    use_tensorboard: bool = False,
    use_csv: bool = True,
    verbose: bool = True,
) -> Logger:
    """Factory for the default RICE logger."""
    return Logger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_tensorboard=use_tensorboard,
        use_csv=use_csv,
        verbose=verbose,
    )


def log_system_info(logger: Optional[Logger] = None) -> Dict[str, str]:
    """Capture basic system information useful for reproducibility logs."""
    info: Dict[str, str] = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
    except Exception:
        info["torch_version"] = "unknown"
        info["cuda_available"] = "unknown"

    try:
        import stable_baselines3 as sb3

        info["sb3_version"] = sb3.__version__
    except Exception:
        info["sb3_version"] = "unknown"

    try:
        import gymnasium as gym

        info["gymnasium_version"] = gym.__version__
    except Exception:
        info["gymnasium_version"] = "unknown"

    if logger is not None:
        logger.log_text("system_info", "\n".join(f"{k}: {v}" for k, v in info.items()))

    return info
