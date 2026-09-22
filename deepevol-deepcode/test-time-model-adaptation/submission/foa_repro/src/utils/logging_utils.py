"""Logging utilities for the FOA reproduction.

This module is pure glue (no paper-specified math).  It centralises the
logging configuration used by every runner script in ``scripts/`` and every
method module in ``src/``:

* :func:`setup_logging` / :func:`configure_logging` -- one-shot configuration of
  the root (or a named) logger, optionally writing to a file as well as the
  console.
* :func:`get_logger` -- retrieve a namespaced logger.
* :func:`log_config` -- pretty-print a (nested) YAML config for the record.
* :func:`format_table` / :func:`log_table` -- tiny dependency-free table
  formatter used for the accuracy / ECE comparison tables (paper Tables 2/3/5/
  9/16/17).
* :class:`ResultsWriter` -- JSON result serialisation helper with the same
  payload shape the runner scripts emit (``{"config": ..., "results": ...}``).
* :func:`log_dict` / :func:`section` -- small conveniences.

Only the Python standard library is used, so this module can never break a run
because of a missing optional dependency.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_DATE_FORMAT",
    "LOG_LEVELS",
    "setup_logging",
    "configure_logging",
    "get_logger",
    "set_log_level",
    "log_config",
    "log_dict",
    "format_table",
    "log_table",
    "section",
    "banner",
    "format_seconds",
    "ResultsWriter",
    "save_results_json",
    "load_results_json",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical log line format: ``2026-01-01 00:00:00 | INFO     | foa.run | msg``
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

#: Timestamp format used by :data:`DEFAULT_LOG_FORMAT`.
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Accepted (string) log level names, lower-cased.
LOG_LEVELS = ("critical", "error", "warning", "info", "debug", "notset")

#: Registry of already-configured logger names -> their handlers.
_CONFIGURED: Dict[str, logging.Logger] = {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _coerce_level(level: Any) -> int:
    """Convert ``level`` (int or str) into a :mod:`logging` level constant."""
    if isinstance(level, int):
        return level
    if level is None:
        return logging.INFO
    name = str(level).strip().upper()
    if name == "WARN":
        name = "WARNING"
    value = getattr(logging, name, None)
    if isinstance(value, int):
        return value
    raise ValueError(
        f"Unknown log level {level!r}; expected one of {LOG_LEVELS} or an int."
    )


def setup_logging(
    level: Any = "info",
    log_file: Optional[str] = None,
    name: Optional[str] = None,
    *,
    quiet: bool = False,
    force: bool = False,
    fmt: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """Configure console (and optional file) logging and return a logger.

    Parameters
    ----------
    level:
        Log level, either an int or a string such as ``"info"`` / ``"DEBUG"``.
    log_file:
        Optional path; when given, a :class:`logging.FileHandler` is attached and
        the parent directory is created if necessary.
    name:
        Logger name.  ``None`` configures the root logger (so every module's
        logger propagates to it).
    quiet:
        Force ``WARNING`` level regardless of ``level`` (used by the ``--quiet``
        CLI flag).
    force:
        Re-configure even if this logger was configured before (handlers are
        removed first).
    fmt, datefmt:
        Formatting strings for the handlers.

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    resolved = logging.WARNING if quiet else _coerce_level(level)
    logger = logging.getLogger(name)

    key = name or "__root__"
    if key in _CONFIGURED and not force:
        # Already configured: just align the level so repeated setup() calls
        # from the runner scripts behave intuitively.
        _set_level(logger, resolved)
        return logger

    if force or key in _CONFIGURED:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover - defensive
                pass

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setLevel(resolved)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file:
        directory = os.path.dirname(os.path.abspath(log_file))
        if directory:
            os.makedirs(directory, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(resolved)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _set_level(logger, resolved)
    # Avoid duplicated lines when the root logger is configured but child loggers
    # also carry a console handler.
    logger.propagate = False
    _CONFIGURED[key] = logger
    return logger


def _set_level(logger: logging.Logger, level: int) -> None:
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


# Friendly aliases -----------------------------------------------------------
configure_logging = setup_logging


def get_logger(name: str = "foa", level: Any = None) -> logging.Logger:
    """Return a namespaced logger, configuring it lazily on first use.

    Parameters
    ----------
    name:
        Logger name, e.g. ``"foa.run_foa"``.
    level:
        Optional level override.
    """
    logger = logging.getLogger(name)
    if level is not None:
        _set_level(logger, _coerce_level(level))
        return logger
    if not logger.handlers and name not in _CONFIGURED:
        setup_logging(level="info", name=name)
    return logger


def set_log_level(level: Any, name: Optional[str] = None) -> None:
    """Change the level of a (previously configured) logger at runtime."""
    logger = logging.getLogger(name)
    _set_level(logger, _coerce_level(level))


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------


def _flatten(obj: Any, prefix: str = "", sep: str = ".") -> List[tuple]:
    """Flatten a nested mapping into ``[(dotted_key, value), ...]``."""
    items: List[tuple] = []
    if isinstance(obj, Mapping):
        for key in obj:
            value = obj[key]
            path = f"{prefix}{sep}{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                items.extend(_flatten(value, path, sep))
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], Mapping):
                # lists of dicts (e.g. ablation grids) are summarised by index
                for idx, element in enumerate(value):
                    items.extend(_flatten(element, f"{path}[{idx}]", sep))
            else:
                items.append((path, value))
    else:
        items.append((prefix, obj))
    return items


def log_dict(
    logger: logging.Logger,
    data: Mapping[str, Any],
    title: Optional[str] = None,
    *,
    level: int = logging.INFO,
    indent: int = 2,
    max_items: Optional[int] = None,
) -> None:
    """Log a nested mapping one ``dotted.key = value`` per line."""
    if title:
        logger.log(level, "%s", title)
    pairs = _flatten(data)
    if max_items is not None:
        pairs = pairs[:max_items]
    pad = " " * indent
    width = max((len(k) for k, _ in pairs), default=0)
    for key, value in pairs:
        logger.log(level, "%s%-*s : %s", pad, width, key, value)


def log_config(
    logger: logging.Logger,
    cfg: Any,
    title: str = "Configuration",
    *,
    level: int = logging.INFO,
) -> None:
    """Log the active configuration (nested dict / ``Config`` / YAML tree)."""
    data: Any = cfg
    if hasattr(cfg, "items"):
        try:
            data = dict(cfg)
        except Exception:  # pragma: no cover - defensive
            data = cfg
    if not isinstance(data, Mapping):
        logger.log(level, "%s: %r", title, data)
        return
    log_dict(logger, data, title=f"=== {title} ===", level=level)


def section(logger: logging.Logger, title: str, char: str = "=", width: int = 72) -> None:
    """Log a section separator, e.g. ``==== ImageNet-C: gaussian_noise ====``."""
    title = f" {title} "
    pad = max(width - len(title), 0)
    left = pad // 2
    right = pad - left
    logger.info("%s%s%s%s%s", char * left, char * (0), "", title, char * right)


def banner(
    logger: logging.Logger,
    title: str,
    subtitle: Optional[str] = None,
    *,
    width: int = 72,
) -> None:
    """Log a boxed banner used at the start of a run."""
    logger.info("%s", "=" * width)
    logger.info("%s", title.center(width))
    if subtitle:
        logger.info("%s", subtitle.center(width))
    logger.info("%s", "=" * width)


def format_seconds(seconds: float) -> str:
    """Human readable duration (``1h 03m 12s`` / ``12.3s``)."""
    if seconds != seconds or seconds < 0:  # NaN / negative guard
        return "n/a"
    seconds = float(seconds)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    return f"{minutes}m {sec:02d}s"


def _format_cell(value: Any, width: Optional[int] = None) -> str:
    if value is None:
        text = "-"
    elif isinstance(value, float):
        text = f"{value:.2f}"
    else:
        text = str(value)
    if width is not None:
        text = text[:width]
    return text


def format_table(
    rows: Sequence[Sequence[Any]],
    headers: Optional[Sequence[str]] = None,
    *,
    sep: str = " | ",
    header_sep: str = "-",
    align: str = "right",
) -> str:
    """Return a simple ASCII table as a string (no third-party dependency).

    ``rows`` may be a sequence of sequences or a sequence of mappings (in which
    case the header order is derived from the keys of the first mapping).
    """
    normalised: List[List[str]] = []
    if rows and isinstance(rows[0], Mapping):
        keys: List[str] = list(headers) if headers else list(rows[0].keys())
        headers = keys
        for row in rows:  # type: ignore[assignment]
            normalised.append([_format_cell(row.get(k)) for k in keys])
    else:
        for row in rows:
            if isinstance(row, (list, tuple)):
                normalised.append([_format_cell(v) for v in row])
            else:
                normalised.append([_format_cell(row)])

    header_cells = [str(h) for h in headers] if headers else []
    n_cols = max([len(header_cells)] + [len(r) for r in normalised] or [0])
    if n_cols == 0:
        return ""

    widths = [0] * n_cols
    for idx in range(n_cols):
        if idx < len(header_cells):
            widths[idx] = max(widths[idx], len(header_cells[idx]))
        for row in normalised:
            if idx < len(row):
                widths[idx] = max(widths[idx], len(row[idx]))

    def _fmt_row(cells: Sequence[str]) -> str:
        out = []
        for idx in range(n_cols):
            cell = cells[idx] if idx < len(cells) else ""
            out.append(cell.ljust(widths[idx]) if align == "left" else cell.rjust(widths[idx]))
        return sep.join(out).rstrip()

    lines: List[str] = []
    if header_cells:
        lines.append(_fmt_row(header_cells))
        lines.append(header_sep * len(lines[0]))
    lines.extend(_fmt_row(row) for row in normalised)
    return "\n".join(lines)


def log_table(
    logger: logging.Logger,
    rows: Sequence[Sequence[Any]],
    headers: Optional[Sequence[str]] = None,
    *,
    title: Optional[str] = None,
    level: int = logging.INFO,
) -> str:
    """Format :func:`format_table` and log it line by line; return the text."""
    if title:
        logger.log(level, "%s", title)
    text = format_table(rows, headers)
    for line in text.splitlines():
        logger.log(level, "%s", line)
    return text


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    """Fallback encoder for numpy scalars / tensors without importing them."""
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            pass
    return str(value)


def save_results_json(payload: Any, path: str, *, indent: int = 2) -> str:
    """Write ``payload`` as JSON, creating parent directories as needed."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, default=_json_default)
    return path


def load_results_json(path: str) -> Any:
    """Read a JSON result file produced by :func:`save_results_json`."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class ResultsWriter:
    """Accumulate named results and dump them to JSON at the end of a run.

    The payload shape mirrors what the runner scripts emit so downstream
    reporting code can consume any of them uniformly::

        {"config": {...}, "results": {"<name>": {...}}, "meta": {...}}
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        config: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        **meta: Any,
    ) -> None:
        self.output_path = output_path
        self.logger = logger or get_logger("foa.results")
        self.results: Dict[str, Any] = {}
        self.meta: Dict[str, Any] = dict(meta)
        self.config: Any = self._to_plain(config) if config is not None else None
        self.started_at = time.time()

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _to_plain(obj: Any) -> Any:
        """Best-effort conversion of config objects into plain containers."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if hasattr(obj, "items"):
            try:
                return {k: ResultsWriter._to_plain(v) for k, v in dict(obj).items()}
            except Exception:  # pragma: no cover - defensive
                return str(obj)
        if isinstance(obj, (list, tuple)):
            return [ResultsWriter._to_plain(v) for v in obj]
        return obj

    # -- API --------------------------------------------------------------
    def add(self, name: str, result: Any) -> Any:
        """Record ``result`` under ``name`` and return it unchanged."""
        self.results[name] = self._to_plain(result)
        return result

    update = add

    def set_config(self, config: Any) -> None:
        self.config = self._to_plain(config)

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.config is not None:
            payload["config"] = self.config
        payload["results"] = self.results
        meta = dict(self.meta)
        meta.setdefault("wall_clock_s", round(time.time() - self.started_at, 3))
        payload["meta"] = meta
        return payload

    def save(self, path: Optional[str] = None) -> Optional[str]:
        target = path or self.output_path
        if not target:
            return None
        save_results_json(self.to_payload(), target)
        self.logger.info("Wrote results to %s", target)
        return target

    def log_summary(self, title: str = "Results") -> None:
        rows: List[List[Any]] = []
        for name, value in self.results.items():
            if isinstance(value, Mapping):
                accuracy = value.get("accuracy")
                ece = value.get("ece")
                rows.append([name, accuracy, ece])
            else:
                rows.append([name, value, None])
        log_table(
            self.logger,
            rows,
            headers=["method", "accuracy", "ece"],
            title=f"=== {title} ===",
        )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.results)

    def __contains__(self, name: object) -> bool:  # pragma: no cover - trivial
        return name in self.results


# ---------------------------------------------------------------------------
# Indented log helper for nested adaptation loops
# ---------------------------------------------------------------------------


class IndentFormatter(logging.Formatter):
    """Formatter adding an indent level controlled by ``record.indent``."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover
        indent = getattr(record, "indent", 0)
        return (" " * (2 * indent)) + super().format(record)


def log_iterable(
    logger: logging.Logger,
    values: Iterable[Any],
    title: str,
    *,
    level: int = logging.INFO,
    max_items: int = 40,
) -> None:
    """Log an iterable (e.g. the 15 ImageNet-C corruption names) compactly."""
    items = list(values)
    shown = items[:max_items]
    suffix = f" (+{len(items) - len(shown)} more)" if len(items) > len(shown) else ""
    logger.log(level, "%s (%d): %s%s", title, len(items), ", ".join(map(str, shown)), suffix)
