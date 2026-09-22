"""Logging utilities for the FRE reproduction.

This module is deliberately lightweight (standard library + optional
``numpy``/``tensorboard``): the training loop in :mod:`fre.fre.trainer` writes a
metric dict every ``log_interval`` steps, and the evaluation scripts need to
persist per-seed / per-task results in a machine readable way so that
:mod:`fre.utils.normalization` can aggregate them into the Table 1 / Table 4
rows.

The paper (:math:`\\S5`) reports ``mean ± std`` over 5 seeds with 20 evaluation
rollouts each; nothing in the paper constrains how logs are written, so this is
a sensible-default utility module.

Public API
----------
* :class:`MetricLogger` -- console + JSONL metric logger with an epoch/step
  counter and an optional TensorBoard writer.
* :class:`AverageMeter` -- running scalar average (kept here as well as in the
  trainer for convenience; both are simple value trackers).
* :class:`TableLogger` -- accumulates rows (e.g. Table 1) and renders markdown.
* :func:`get_logger`, :func:`configure_logging` -- stdlib ``logging`` helpers.
* :func:`make_run_dir`, :func:`save_json`, :func:`load_json`,
  :func:`save_csv`, :func:`save_numpy`.
* :func:`format_metrics`, :func:`flatten_dict`, :func:`as_scalar`.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # numpy is used everywhere else in the project but keep the import soft
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a hard dependency in practice
    _np = None  # type: ignore

__all__ = [
    "MetricLogger",
    "AverageMeter",
    "TableLogger",
    "get_logger",
    "configure_logging",
    "make_run_dir",
    "save_json",
    "load_json",
    "save_csv",
    "save_numpy",
    "format_metrics",
    "flatten_dict",
    "as_scalar",
    "Timer",
]


# ---------------------------------------------------------------------------
# stdlib logging helpers
# ---------------------------------------------------------------------------

_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = _LOG_FORMAT,
    datefmt: str = _DATE_FORMAT,
    force: bool = False,
) -> None:
    """Configure the root logger once.

    Parameters
    ----------
    level:
        Logging level (``logging.INFO`` by default).
    log_file:
        Optional path to also stream the log to a file.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger()
    root.setLevel(level)
    # Remove pre-existing handlers when force-reconfiguring.
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(formatter)
    stream.setLevel(level)
    root.addHandler(stream)

    if log_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str = "fre", level: Optional[int] = None) -> logging.Logger:
    """Return a module logger, configuring the root logger on first use."""
    configure_logging()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# scalar helpers
# ---------------------------------------------------------------------------

def as_scalar(value: Any) -> Any:
    """Best-effort conversion of a tensor / array / scalar to a Python number.

    Returns the value unchanged when the conversion is not possible (e.g. a
    string), so that :func:`flatten_dict` can tolerate mixed payloads.
    """
    if isinstance(value, (int, float, bool, str)):
        return value

    if value is None:
        return None

    # torch tensors expose ``detach``/``cpu``/``item``.
    try:
        detached = value.detach()  # type: ignore[attr-defined]
        if hasattr(detached, "numel") and detached.numel() == 1:  # type: ignore[attr-defined]
            return float(detached.item())  # type: ignore[attr-defined]
    except Exception:
        pass

    if _np is not None and isinstance(value, _np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return value

    try:
        if hasattr(value, "item"):
            return float(value.item())  # type: ignore[call-arg]
    except Exception:
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def flatten_dict(
    d: Mapping[str, Any], prefix: str = "", sep: str = "/", max_depth: int = 6,
) -> Dict[str, Any]:
    """Flatten a nested metric dict into a ``prefix/key`` mapping.

    Scalars are converted with :func:`as_scalar`; nested dicts are recursed
    into up to ``max_depth`` levels.
    """
    flat: Dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, Mapping) and max_depth > 0:
            flat.update(flatten_dict(value, name, sep=sep, max_depth=max_depth - 1))
        else:
            flat[name] = as_scalar(value)
    return flat


def format_metrics(d: Mapping[str, Any], precision: int = 4) -> str:
    """Format a (possibly nested) metric dict as a single log line."""
    parts: List[str] = []
    for key, value in flatten_dict(d).items():
        value = as_scalar(value)
        if isinstance(value, float):
            parts.append(f"{key}={value:.{precision}g}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


class Timer:
    """Minimal wall-clock context manager (``with Timer() as t: ...``)."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.time() - self.start_time

    def reset(self) -> None:
        self.start_time = time.time()
        self.elapsed = 0.0


class AverageMeter:
    """Tracks the running mean of a scalar."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.value = 0.0

    def update(self, value: Any, n: int = 1) -> float:
        value = as_scalar(value)
        if not isinstance(value, (int, float)):
            return self.value
        self.sum += float(value) * n
        self.count += n
        self.value = self.sum / max(self.count, 1)
        return self.value

    @property
    def mean(self) -> float:
        return self.value

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"AverageMeter(name={self.name!r}, value={self.value:.6g}, n={self.count})"


# ---------------------------------------------------------------------------
# metric logger
# ---------------------------------------------------------------------------

class MetricLogger:
    """Console + JSONL metric logger with optional TensorBoard support.

    Parameters
    ----------
    log_dir:
        Directory where ``metrics.jsonl`` (and the TensorBoard event files) are
        written.  ``None`` disables file logging.
    name:
        Logger name used for the stdlib logger.
    use_tensorboard:
        Try to create a ``SummaryWriter``; failures are silently ignored so the
        code still runs without the ``tensorboard`` package installed.
    verbose:
        Also print each logged metric line to stdout.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        name: str = "fre",
        use_tensorboard: bool = False,
        verbose: bool = True,
        flush_every: int = 100,
    ) -> None:
        self.log_dir = log_dir
        self.verbose = verbose
        self.flush_every = max(int(flush_every), 1)
        self.step = 0
        self.start_time = time.time()
        self.logger = get_logger(name)

        self._jsonl_path: Optional[str] = None
        self._jsonl_file = None
        self._since_flush = 0

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self._jsonl_path = os.path.join(log_dir, "metrics.jsonl")
            self._jsonl_file = open(self._jsonl_path, "a", buffering=1)

        self.writer = None
        if use_tensorboard and log_dir is not None:
            try:  # pragma: no cover - optional dependency
                from torch.utils.tensorboard import SummaryWriter  # type: ignore

                self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
            except Exception:
                self.writer = None

    # -- counters ---------------------------------------------------------
    def set_step(self, step: int) -> None:
        self.step = int(step)

    def increment_step(self, amount: int = 1) -> int:
        self.step += int(amount)
        return self.step

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    # -- logging ----------------------------------------------------------
    def log(
        self,
        metrics: Mapping[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
        console: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Log a metric dict at ``step`` (defaults to the internal counter)."""
        if step is None:
            step = self.step
        step = int(step)

        flat = flatten_dict(metrics)
        if prefix:
            flat = {f"{prefix}/{k}": v for k, v in flat.items()}

        record: Dict[str, Any] = {"step": step, "wall_time": self.elapsed}
        record.update(flat)

        if self._jsonl_file is not None:
            try:
                self._jsonl_file.write(json.dumps(record) + "\n")
                self._since_flush += 1
                if self._since_flush >= self.flush_every:
                    self._jsonl_file.flush()
                    self._since_flush = 0
            except Exception:  # pragma: no cover - never crash training on I/O
                pass

        if self.writer is not None:  # pragma: no cover - optional dependency
            for key, value in flat.items():
                if isinstance(value, (int, float, bool)):
                    try:
                        self.writer.add_scalar(key, float(value), step)
                    except Exception:
                        pass

        show = self.verbose if console is None else console
        if show:
            self.logger.info("step=%d %s", step, format_metrics(flat))

        return record

    # alias used by some call sites
    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        return self.log(metrics, step=step, **kwargs)

    def log_hyperparameters(self, hparams: Mapping[str, Any]) -> None:
        """Persist a hyperparameter dict as ``hparams.json`` next to the logs."""
        if self.log_dir is None:
            return
        save_json(os.path.join(self.log_dir, "hparams.json"), dict(hparams))
        if self.writer is not None:  # pragma: no cover - optional dependency
            try:
                self.writer.add_text("hparams", json.dumps(dict(hparams), indent=2, default=str))
            except Exception:
                pass

    def info(self, message: str, *args: Any) -> None:
        self.logger.info(message, *args)

    def warning(self, message: str, *args: Any) -> None:
        self.logger.warning(message, *args)

    def close(self) -> None:
        if self._jsonl_file is not None:
            try:
                self._jsonl_file.flush()
                self._jsonl_file.close()
            except Exception:  # pragma: no cover
                pass
            self._jsonl_file = None
        if self.writer is not None:  # pragma: no cover - optional dependency
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# table logger (Table 1 / Table 4 style accumulation)
# ---------------------------------------------------------------------------

@dataclass
class TableLogger:
    """Accumulates ``row -> column -> value`` cells and renders a table.

    Used by the evaluation scripts to collect per-task scores and print them in
    the same layout as the paper's Table 1:

    >>> t = TableLogger()
    >>> t.add("ant-goal-reaching", {"FRE": 48.8, "FB": 20.0})
    >>> print(t.render())  # doctest: +SKIP
    """

    rows: "MutableMapping[str, Dict[str, Any]]" = field(default_factory=dict)
    columns: List[str] = field(default_factory=list)

    def add(self, row: str, values: Mapping[str, Any]) -> None:
        bucket = self.rows.setdefault(str(row), {})
        for key, value in values.items():
            key = str(key)
            if key not in self.columns:
                self.columns.append(key)
            bucket[key] = as_scalar(value)

    def set(self, row: str, column: str, value: Any) -> None:
        self.add(row, {column: value})

    def get(self, row: str, column: str, default: Any = None) -> Any:
        return self.rows.get(str(row), {}).get(str(column), default)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {row: dict(cells) for row, cells in self.rows.items()}

    def render(
        self,
        digits: int = 1,
        title: Optional[str] = None,
        row_order: Optional[Sequence[str]] = None,
    ) -> str:
        order = list(row_order) if row_order is not None else list(self.rows.keys())
        header = ["task"] + self.columns
        widths = [max(len(str(h)), 12 if i == 0 else len(str(h))) for i, h in enumerate(header)]
        for row in order:
            cells = self.rows.get(row, {})
            widths[0] = max(widths[0], len(str(row)))
            for j, col in enumerate(self.columns, start=1):
                widths[j] = max(widths[j], len(_format_cell(cells.get(col), digits)))

        def fmt_line(values: Sequence[str]) -> str:
            return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

        sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
        lines: List[str] = []
        if title:
            lines.append(f"### {title}")
        lines.append(fmt_line(header))
        lines.append(sep)
        for row in order:
            cells = self.rows.get(row, {})
            lines.append(fmt_line([str(row)] + [_format_cell(cells.get(col), digits) for col in self.columns]))
        return "\n".join(lines)

    def markdown(self, digits: int = 1, title: Optional[str] = None, **kwargs: Any) -> str:
        return self.render(digits=digits, title=title, **kwargs)


def _format_cell(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return f"{float(value[0]):.{digits}f}±{float(value[1]):.{digits}f}"
        except (TypeError, ValueError):
            return str(tuple(value))
    return str(value)


# ---------------------------------------------------------------------------
# filesystem helpers
# ---------------------------------------------------------------------------

def make_run_dir(
    root: str = "runs",
    name: Optional[str] = None,
    seed: Optional[int] = None,
    timestamp: bool = True,
    subdirs: Optional[Sequence[str]] = None,
) -> str:
    """Create (and return) a unique run directory under ``root``.

    ``runs/<timestamp>_<name>_seed<seed>`` by default; optional ``subdirs``
    (e.g. ``["checkpoints", "eval"]``) are created inside it.
    """
    parts: List[str] = []
    if timestamp:
        parts.append(time.strftime("%Y%m%d-%H%M%S"))
    if name:
        parts.append(str(name))
    if seed is not None:
        parts.append(f"seed{int(seed)}")
    dirname = "_".join(parts) if parts else "run"

    path = os.path.join(root, dirname)
    # Avoid clobbering an existing run directory with the same second-granularity name.
    suffix = 1
    unique = path
    while os.path.exists(unique):
        unique = f"{path}_{suffix}"
        suffix += 1

    os.makedirs(unique, exist_ok=True)
    for sub in subdirs or ():
        os.makedirs(os.path.join(unique, sub), exist_ok=True)
    return unique


def save_json(path: str, payload: Any, indent: int = 2) -> str:
    """Write ``payload`` as JSON, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=indent, default=_json_default)
    return path


def load_json(path: str, default: Any = None) -> Any:
    """Read JSON from ``path``; return ``default`` when the file is missing."""
    if not os.path.exists(path):
        return default
    with open(path, "r") as handle:
        return json.load(handle)


def _json_default(obj: Any) -> Any:
    if _np is not None and isinstance(obj, (_np.ndarray, _np.floating, _np.integer)):
        return obj.tolist() if isinstance(obj, _np.ndarray) else obj.item()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def save_csv(
    path: str,
    rows: Iterable[Mapping[str, Any]],
    columns: Optional[Sequence[str]] = None,
) -> str:
    """Write an iterable of flat row dicts to CSV."""
    rows = [dict(row) for row in rows]
    if not columns:
        columns = []
        for row in rows:
            for key in row.keys():
                if key not in columns:
                    columns.append(key)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    return path


def save_numpy(path: str, **arrays: Any) -> str:
    """Save named arrays to a ``.npz`` file (requires numpy)."""
    if _np is None:  # pragma: no cover
        raise RuntimeError("numpy is required for save_numpy")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    _np.savez(path, **arrays)
    return path


# ---------------------------------------------------------------------------
# convenience aggregate used by the eval scripts
# ---------------------------------------------------------------------------

def summarize_seeds(
    per_seed_results: Sequence[Mapping[str, Any]],
    keys: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate a list of per-seed result dicts into ``{key: {mean, std}}``.

    This is a thin wrapper so evaluation drivers do not need to import
    :mod:`fre.utils.normalization` just to print a summary.
    """
    summary: Dict[str, Dict[str, float]] = {}
    if not per_seed_results:
        return summary

    if keys is None:
        keys = []
        for result in per_seed_results:
            for key, value in result.items():
                if isinstance(as_scalar(value), (int, float)) and key not in keys:
                    keys.append(key)

    for key in keys:
        values: List[float] = []
        for result in per_seed_results:
            value = as_scalar(result.get(key))
            if isinstance(value, (int, float)):
                values.append(float(value))
        if not values:
            continue
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = var ** 0.5
        else:
            std = 0.0
        summary[str(key)] = {"mean": mean, "std": std, "num_seeds": float(len(values))}
    return summary


def result_row(score: float, std: float, digits: int = 1) -> Tuple[float, float]:
    """Round a ``(mean, std)`` pair for reporting (used by Table 1 printing)."""
    return (round(float(score), digits), round(float(std), digits))
