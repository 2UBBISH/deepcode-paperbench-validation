"""Logging, metric-tracking and seeding utilities for the FRE reproduction.

This module intentionally has *no* hard dependency on ``torch`` so that it can be
imported by data-processing scripts / unit tests that only require ``numpy``.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict, deque
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "set_seed",
    "seed_everything",
    "get_logger",
    "configure_logging",
    "Logger",
    "MetricTracker",
    "RunningMeanStd",
    "CSVLogger",
    "JsonlLogger",
    "write_json",
    "read_json",
    "format_time",
    "progress",
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_DATE_FORMAT",
]

DEFAULT_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%H:%M:%S"

_CONFIGURED = False


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------
def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed python / numpy / torch (if available) RNGs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # pragma: no cover - torch optional
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover
        pass
    return seed


#: Alias used across the code base.
set_seed = seed_everything


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
def configure_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    global _CONFIGURED
    root = logging.getLogger("fre")
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT))
        root.addHandler(handler)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT))
        root.addHandler(fh)
    _CONFIGURED = True


def get_logger(
    name: str = "fre",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Return a namespaced logger, configuring the root on first use."""
    if not _CONFIGURED:
        configure_logging(level=level, log_file=log_file)
    logger = logging.getLogger(name if name.startswith("fre") else f"fre.{name}")
    logger.setLevel(level)
    return logger


class Logger:
    """Minimal logging facade with ``info/metric/warning`` helpers.

    ``metric`` prints key/value pairs on a single line, matching the style used
    in the FRE training scripts.
    """

    def __init__(
        self,
        name: str = "fre",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        self.logger = get_logger(name, level=level, log_file=log_file)
        self.verbose = verbose

    # passthroughs ---------------------------------------------------------
    def info(self, msg: str, *args: Any) -> None:
        self.logger.info(msg, *args)

    def debug(self, msg: str, *args: Any) -> None:
        self.logger.debug(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self.logger.warning(msg, *args)

    warn = warning

    def error(self, msg: str, *args: Any) -> None:
        self.logger.error(msg, *args)

    # structured -----------------------------------------------------------
    def metric(self, step: Optional[int] = None, **metrics: Any) -> None:
        if not self.verbose:
            return
        parts = []
        if step is not None:
            parts.append(f"step={int(step)}")
        for key, value in metrics.items():
            parts.append(f"{key}={_fmt(value)}")
        self.logger.info(" | ".join(parts))

    def key_values(self, prefix: str, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            self.logger.info("%s.%s = %s", prefix, key, _fmt(value))


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, np.floating):
        return f"{float(value):.4f}"
    if isinstance(value, np.integer):
        return str(int(value))
    return str(value)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
class RunningMeanStd:
    """Numerically stable running mean/variance estimator (Welford)."""

    def __init__(self, shape: Sequence[int] = (), eps: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(eps)

    def update(self, values: np.ndarray) -> "RunningMeanStd":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == self.mean.ndim:
            values = values[np.newaxis]
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)
        batch_count = values.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
        return self

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state.get("count", 1e-4))


class MetricTracker:
    """Tracks running averages (and last values) for a set of named metrics."""

    def __init__(self, names: Optional[Iterable[str]] = None, window: int = 100) -> None:
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self._last: Dict[str, Any] = {}
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        if names:
            for name in names:
                self._sums[name] = 0.0
                self._counts[name] = 0

    def update(self, values: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        items: Dict[str, Any] = {}
        if values:
            items.update(values)
        items.update(kwargs)
        for key, value in items.items():
            arr = np.asarray(value)
            scalar = float(arr.mean()) if arr.size else float(value)
            self._sums[key] += scalar
            self._counts[key] += 1
            self._last[key] = value
            self._history[key].append(scalar)

    def mean(self, key: Optional[str] = None) -> Any:
        if key is None:
            return {k: self.mean(k) for k in self._sums}
        if not self._counts.get(key):
            return float("nan")
        return self._sums[key] / self._counts[key]

    def last(self, key: Optional[str] = None) -> Any:
        if key is None:
            return dict(self._last)
        return self._last.get(key)

    def recent(self, key: str) -> float:
        hist = self._history.get(key)
        if not hist:
            return float("nan")
        return float(np.mean(hist))

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()
        self._last.clear()
        self._history.clear()

    def as_dict(self, recent: bool = False) -> Dict[str, float]:
        keys = set(self._sums) | set(self._history)
        out = {}
        for key in keys:
            out[key] = self.recent(key) if recent else self.mean(key)
        return out

    def __contains__(self, key: str) -> bool:
        return key in self._sums or key in self._history

    def __getitem__(self, key: str) -> float:
        return self.mean(key)


# ---------------------------------------------------------------------------
# structured output
# ---------------------------------------------------------------------------
class CSVLogger:
    """Append-only CSV writer that discovers its fieldnames from the first row."""

    def __init__(self, path: str, fieldnames: Optional[Sequence[str]] = None) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fieldnames = list(fieldnames) if fieldnames else None
        self._file = None
        self._writer = None

    def log(self, row: Mapping[str, Any]) -> None:
        if self._writer is None:
            if self._fieldnames is None:
                self._fieldnames = list(row.keys())
            self._file = open(self.path, "w", newline="")
            self._writer = _csv_writer(self._file, self._fieldnames)
            self._writer.writeheader()
        self._writer.writerow({k: row.get(k, "") for k in self._fieldnames})
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _csv_writer(fh, fieldnames):
    import csv

    return csv.DictWriter(fh, fieldnames=fieldnames)


class JsonlLogger:
    """Writes newline-delimited JSON records (one dict per line)."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "a")

    def log(self, row: Mapping[str, Any]) -> None:
        self._fh.write(json.dumps(_jsonable(row)) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float):
        return obj
    try:  # tensors etc.
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:  # pragma: no cover
        pass
    return obj


def write_json(path: str, obj: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_jsonable(obj), fh, indent=2)
    return path


def read_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
def format_time(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{sec:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m"


class _NullProgress:
    def __init__(self, iterable: Iterable, **kwargs: Any) -> None:
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def update(self, n: int = 1) -> None:
        pass

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_description(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def progress(iterable: Optional[Iterable] = None, total: Optional[int] = None, desc: Optional[str] = None, **kwargs: Any):
    """``tqdm`` if installed, otherwise a no-op shim."""
    try:
        from tqdm.auto import tqdm as _tqdm

        return _tqdm(iterable, total=total, desc=desc, **kwargs)
    except Exception:  # pragma: no cover
        if iterable is None:
            return _NullProgress(range(total or 0), **kwargs)
        return _NullProgress(iterable, **kwargs)


if __name__ == "__main__":  # pragma: no cover - smoke test
    log = Logger("fre.demo")
    log.metric(step=1, loss=0.1234, recon=0.5)
    tracker = MetricTracker()
    for i in range(10):
        tracker.update(loss=i, acc=1.0 - 0.1 * i)
    print("means:", tracker.as_dict())
    rms = RunningMeanStd(shape=(3,))
    rms.update(np.random.randn(100, 3))
    print("mean/std:", rms.mean, rms.std)
    print("time:", format_time(3725))
