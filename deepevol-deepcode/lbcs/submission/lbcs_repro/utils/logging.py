"""Logging utilities for the LBCS reproduction.

This module is part of the experiment *glue layer*: it contains no
paper-specific formula.  It provides

* :func:`setup_logging` / :func:`get_logger` -- consistent console (+ optional
  file) logging for the whole package,
* :class:`MetricTracker` -- a small container that records ``f1``/``f2`` (and
  arbitrary) values per outer iteration so the Figure 1 / Table 1 style curves
  and tables can be logged and dumped,
* :class:`ExperimentLogger` -- writes JSON / JSONL / TXT artifacts under a
  results directory, mirroring the artifact layout used by the experiment
  drivers (``results/<experiment>/...``),
* :func:`format_value_table` -- a tiny text-table renderer for logging summary
  tables without pulling in pandas,
* :func:`describe_config` -- compact, secret-free config dump for the log.

PyTorch / NumPy are treated as *soft* dependencies: nothing here requires a
GPU and the module works in a numpy-only environment.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # numpy is effectively always available, but keep the import guarded
    import numpy as _np

    _NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover - extremely unlikely
    _np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False


LOGGER = logging.getLogger(__name__)

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = "INFO"

#: Name of the package-root logger that :func:`setup_logging` configures.
ROOT_LOGGER_NAME = "lbcs_repro"

DEFAULT_OUTPUT_DIR = "results"

#: Artifact suffixes produced by :meth:`ExperimentLogger.save`.
ARTIFACT_SUFFIXES = (".json", ".jsonl", ".txt", ".csv")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    """Fallback encoder for objects that are not natively JSON serializable."""
    if _NUMPY_AVAILABLE:
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, _np.generic):  # numpy scalar
            return obj.item()
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return obj.to_dict()
        except TypeError:  # pragma: no cover - signature mismatch
            return dict(obj)  # type: ignore[arg-type]
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return obj.tolist()
        except Exception:  # pragma: no cover
            pass
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, (os.PathLike,)):
        return os.fspath(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    return repr(obj)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert *obj* into plain JSON-serializable Python objects."""
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if _NUMPY_AVAILABLE:
        if isinstance(obj, _np.ndarray):
            return to_jsonable(obj.tolist())
        if isinstance(obj, _np.generic):
            return to_jsonable(obj.item())
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return to_jsonable(obj.to_dict())
        except TypeError:  # pragma: no cover
            pass
    return _json_default(obj)


def is_finite(value: Any) -> bool:
    """Return ``True`` when *value* is a finite real number."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Logger configuration
# ---------------------------------------------------------------------------
@dataclass
class LoggingConfig:
    """Container for logging configuration (mirrors ``configs/default.yaml``)."""

    level: str = DEFAULT_LOG_LEVEL
    log_every: int = 0
    to_file: bool = False
    file_name: str = "run.log"
    output_dir: str = DEFAULT_OUTPUT_DIR
    log_format: str = DEFAULT_LOG_FORMAT
    date_format: str = DEFAULT_DATE_FORMAT
    quiet_libraries: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]] = None, **overrides: Any) -> "LoggingConfig":
        """Build a config from a (possibly nested) mapping, ignoring unknown keys."""
        cfg = cls()
        if data:
            unknown = set(data) - set(cls.__dataclass_fields__)
            if unknown:
                LOGGER.debug("Ignoring unknown logging config keys: %s", sorted(unknown))
            for key, value in data.items():
                if key in cls.__dataclass_fields__ and value is not None:
                    setattr(cfg, key, value)
        return cfg.with_overrides(**overrides)

    def with_overrides(self, **overrides: Any) -> "LoggingConfig":
        """Return a copy with the given fields replaced (``None`` is ignored)."""
        data = {k: v for k, v in overrides.items() if v is not None and k in self.__dataclass_fields__}
        if not data:
            return self
        merged = {f: getattr(self, f) for f in self.__dataclass_fields__}
        merged.update(data)
        return LoggingConfig(**merged)

    def to_dict(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    @property
    def log_path(self) -> Optional[str]:
        """Absolute-ish path of the log file, or ``None`` when file logging is off."""
        if not self.to_file:
            return None
        if os.path.isabs(self.file_name):
            return self.file_name
        return os.path.join(self.output_dir, self.file_name)


def _resolve_level(level: Any) -> int:
    """Coerce a level name/number into a ``logging`` level."""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        value = logging.getLevelName(level.upper())
        if isinstance(value, int):
            return value
    return logging.INFO


def setup_logging(
    level: Any = DEFAULT_LOG_LEVEL,
    log_file: Optional[str] = None,
    *,
    output_dir: Optional[str] = None,
    file_name: Optional[str] = None,
    fmt: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
    name: str = ROOT_LOGGER_NAME,
    quiet_libraries: bool = True,
    force: bool = True,
) -> logging.Logger:
    """Configure the package logger and return it.

    Console handler is always added (when *force* is true the configuration is
    rebuilt, so repeated calls do not duplicate handlers).  A file handler is
    added when *log_file* (or *output_dir*/*file_name*) is given.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_resolve_level(level))
    logger.propagate = False

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover
                pass

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    path = log_file
    if path is None and file_name:
        path = os.path.join(output_dir or DEFAULT_OUTPUT_DIR, file_name)
    if path:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if quiet_libraries:
        for noisy in ("matplotlib", "PIL", "urllib3", "filelock", "torchvision"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


def configure_logging(config: Any = None, **overrides: Any) -> logging.Logger:
    """Set up logging from a :class:`LoggingConfig`, mapping or ``None``."""
    if config is None:
        cfg = LoggingConfig.from_dict(None, **overrides)
    elif isinstance(config, LoggingConfig):
        cfg = config.with_overrides(**overrides)
    else:
        cfg = LoggingConfig.from_dict(config, **overrides)

    log_file: Optional[str] = None
    if cfg.to_file:
        log_file = cfg.log_path
    return setup_logging(
        level=cfg.level,
        log_file=log_file,
        output_dir=cfg.output_dir,
        fmt=cfg.log_format,
        datefmt=cfg.date_format,
        quiet_libraries=cfg.quiet_libraries,
    )


def get_logger(name: Optional[str] = None, level: Any = None) -> logging.Logger:
    """Return a namespaced logger under ``lbcs_repro`` (or the root logger)."""
    if not name:
        logger = logging.getLogger(ROOT_LOGGER_NAME)
    elif name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
    if level is not None:
        logger.setLevel(_resolve_level(level))
    return logger


def log_section(logger: logging.Logger, title: str, char: str = "=", width: int = 78) -> None:
    """Log a visually separated section header."""
    logger.info(char * width)
    logger.info(title)
    logger.info(char * width)


def log_kv(logger: logging.Logger, mapping: Mapping[str, Any], prefix: str = "  ") -> None:
    """Log a mapping as ``key : value`` lines."""
    for key, value in mapping.items():
        logger.info("%s%-28s %s", prefix, f"{key}:", value)


def describe_config(config: Any, max_items: int = 60) -> Dict[str, Any]:
    """Return a compact, log-friendly view of a config object/mapping."""
    if config is None:
        return {}
    if hasattr(config, "to_dict") and callable(config.to_dict):
        data: Mapping[str, Any] = config.to_dict()
    elif isinstance(config, Mapping):
        data = config
    else:  # fall back to __dict__ for plain objects
        data = getattr(config, "__dict__", {}) or {}

    out: Dict[str, Any] = {}
    for key, value in list(data.items())[:max_items]:
        if value is None or isinstance(value, (int, float, str, bool, list, tuple)):
            out[key] = value
        elif isinstance(value, Mapping):
            out[key] = dict(list(value.items())[:8])
        else:
            out[key] = type(value).__name__
    return out


def log_config(logger: logging.Logger, config: Any, title: str = "Configuration") -> None:
    """Log a config summary at INFO level."""
    log_section(logger, title, char="-")
    log_kv(logger, describe_config(config))


# ---------------------------------------------------------------------------
# Metric tracking
# ---------------------------------------------------------------------------
@dataclass
class MetricRecord:
    """One logged measurement (typically one outer iteration of Algorithm 1/2)."""

    iteration: int
    metrics: Dict[str, float] = field(default_factory=dict)
    tag: Optional[str] = None
    wall_time: Optional[float] = None

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        return self.metrics.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"iteration": self.iteration}
        data.update({k: v for k, v in self.metrics.items()})
        if self.tag is not None:
            data["tag"] = self.tag
        if self.wall_time is not None:
            data["wall_time"] = self.wall_time
        return data

    def __getitem__(self, key: str) -> Any:
        return self.metrics[key]


class MetricTracker:
    """Accumulates scalar metrics per step and exposes them as curves.

    Designed for the LBCS objective traces: ``tracker.add(1, f1=..., f2=...)``.
    Missing values are simply absent from the corresponding curve, so partially
    logged runs still produce usable plots/tables.
    """

    def __init__(self, keys: Optional[Sequence[str]] = None, log_every: int = 0, logger: Optional[logging.Logger] = None):
        self.keys: List[str] = list(keys) if keys else []
        self.log_every = int(log_every or 0)
        self.logger = logger
        self.records: List[MetricRecord] = []
        self.t0 = time.time()

    # -- recording ---------------------------------------------------------
    def add(self, iteration: Optional[int] = None, tag: Optional[str] = None, **metrics: Any) -> MetricRecord:
        """Record one step of metrics and return the created :class:`MetricRecord`."""
        if iteration is None:
            iteration = len(self.records)
        clean: Dict[str, float] = {}
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, bool):
                clean[key] = float(value)
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not is_finite(numeric):
                continue
            clean[key] = numeric
            if key not in self.keys:
                self.keys.append(key)

        record = MetricRecord(
            iteration=int(iteration),
            metrics=clean,
            tag=tag,
            wall_time=time.time() - self.t0,
        )
        self.records.append(record)

        if self.logger is not None and self.log_every and len(self.records) % self.log_every == 0:
            pretty = " ".join(f"{k}={v:.4f}" for k, v in clean.items())
            self.logger.info("iter %d | %s", record.iteration, pretty)
        return record

    def update(self, values: Mapping[str, Any], iteration: Optional[int] = None, **extra: Any) -> MetricRecord:
        """Convenience wrapper accepting a mapping instead of keyword values."""
        merged: Dict[str, Any] = dict(values)
        merged.update(extra)
        return self.add(iteration=iteration, **merged)

    # -- access ------------------------------------------------------------
    def curve(self, key: str) -> List[float]:
        """Return the ordered values logged for *key*."""
        return [r.metrics[key] for r in self.records if key in r.metrics]

    def iterations(self, key: Optional[str] = None) -> List[int]:
        """Iterations that logged *key* (or all iterations when ``None``)."""
        if key is None:
            return [r.iteration for r in self.records]
        return [r.iteration for r in self.records if key in r.metrics]

    def xy(self, key: str) -> Tuple[List[int], List[float]]:
        """Return ``(iterations, values)`` for *key* — ready for plotting."""
        xs, ys = [], []
        for record in self.records:
            if key in record.metrics:
                xs.append(record.iteration)
                ys.append(record.metrics[key])
        return xs, ys

    def latest(self, key: Optional[str] = None) -> Optional[float]:
        """Most recent value of *key* (or of any metric when ``None``)."""
        for record in reversed(self.records):
            if key is None:
                if record.metrics:
                    return list(record.metrics.values())[-1]
            elif key in record.metrics:
                return record.metrics[key]
        return None

    def best(self, key: str, maximize: bool = False) -> Optional[float]:
        """Best (min by default) value seen for *key*."""
        values = self.curve(key)
        if not values:
            return None
        return max(values) if maximize else min(values)

    def summary(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Per-key first/last/min/max of the tracked curves."""
        out: Dict[str, Dict[str, Optional[float]]] = {}
        for key in self.keys:
            values = self.curve(key)
            if not values:
                out[key] = {"first": None, "last": None, "min": None, "max": None, "count": 0}
                continue
            out[key] = {
                "first": values[0],
                "last": values[-1],
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
        return out

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "keys": list(self.keys),
            "records": [r.to_dict() for r in self.records],
            "curves": {k: self.curve(k) for k in self.keys},
            "iterations": {k: self.iterations(k) for k in self.keys},
            "summary": self.summary(),
        }

    def to_records(self) -> List[Dict[str, Any]]:
        """Flat list of record dicts (JSONL friendly)."""
        return [r.to_dict() for r in self.records]

    def to_rows(self, keys: Optional[Sequence[str]] = None, decimals: int = 4) -> Tuple[List[str], List[List[str]]]:
        """Render the tracker as ``(headers, rows)`` text-table data."""
        keys = list(keys) if keys else list(self.keys)
        headers = ["iter"] + keys
        rows: List[List[str]] = []
        for record in self.records:
            row = [str(record.iteration)]
            for key in keys:
                value = record.metrics.get(key)
                row.append("-" if value is None else f"{value:.{decimals}f}")
            rows.append(row)
        return headers, rows

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, key: str) -> List[float]:
        return self.curve(key)


# ---------------------------------------------------------------------------
# Experiment artifact logger
# ---------------------------------------------------------------------------
def format_value_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    separator: str = " | ",
    decimals: int = 4,
) -> str:
    """Render a plain-text table (no external dependency)."""

    def render(value: Any) -> str:
        if isinstance(value, float):
            if not math.isfinite(value):
                return "-"
            return f"{value:.{decimals}f}"
        if value is None:
            return "-"
        return str(value)

    body = [[render(v) for v in row] for row in rows]
    head = [str(h) for h in headers]
    widths = [len(h) for h in head]
    for row in body:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    def line(cells: Sequence[str]) -> str:
        padded = [cells[i].ljust(widths[i]) if i < len(widths) else cells[i] for i in range(len(cells))]
        return separator.join(padded)

    out = [line(head), separator.join("-" * w for w in widths)]
    out.extend(line(row) for row in body)
    return "\n".join(out)


class ExperimentLogger:
    """Writes experiment artifacts (JSON/JSONL/TXT/CSV) under a results dir.

    The drivers call e.g.::

        elog = ExperimentLogger("table1", output_dir="results/table1", logger=logger)
        elog.log_metrics("k200_eps0.2", {"f1": 1.92, "f2": 190.7})
        elog.save_json("table1", payload)
        elog.save_text("table1", rendered_table)
    """

    def __init__(
        self,
        experiment: str,
        output_dir: Optional[str] = None,
        *,
        logger: Optional[logging.Logger] = None,
        save_artifacts: bool = True,
        seed: Optional[int] = None,
        config: Any = None,
    ):
        self.experiment = experiment
        self.output_dir = output_dir or os.path.join(DEFAULT_OUTPUT_DIR, experiment)
        self.logger = logger or LOGGER
        self.save_artifacts = bool(save_artifacts)
        self.seed = seed
        self.config = config
        self.tracker = MetricTracker(log_every=0, logger=None)
        self.artifacts: Dict[str, str] = {}
        self.t0 = time.time()
        if self.save_artifacts:
            os.makedirs(self.output_dir, exist_ok=True)

    # -- paths -------------------------------------------------------------
    def path(self, name: str, suffix: str = ".json") -> str:
        """Full path for an artifact called *name*."""
        if not os.path.splitext(name)[1]:
            name = f"{name}{suffix}"
        return os.path.join(self.output_dir, name)

    # -- metric logging ----------------------------------------------------
    def log_metrics(self, tag: str, metrics: Mapping[str, Any], iteration: Optional[int] = None) -> MetricRecord:
        """Record metrics with a *tag* (e.g. ``"k200_eps0.2"``)."""
        record = self.tracker.add(iteration=iteration, tag=tag, **dict(metrics))
        self.logger.debug("metrics[%s] %s", tag, record.metrics)
        return record

    def log_curve(self, key: str, values: Sequence[float], tags: Optional[Sequence[str]] = None) -> None:
        """Record a whole curve, one record per value."""
        for i, value in enumerate(values):
            tag = tags[i] if tags is not None and i < len(tags) else None
            self.tracker.add(iteration=i, tag=tag, **{key: value})

    def log_message(self, message: str, level: int = logging.INFO) -> None:
        self.logger.log(level, message)

    # -- artifact writers --------------------------------------------------
    def save_json(self, name: str, payload: Any, indent: int = 2) -> str:
        path = self.path(name, ".json")
        self._write(path, json.dumps(to_jsonable(payload), indent=indent) + "\n")
        return path

    def save_jsonl(self, name: str, records: Iterable[Mapping[str, Any]]) -> str:
        path = self.path(name, ".jsonl")
        lines = [json.dumps(to_jsonable(dict(record))) for record in records]
        self._write(path, "\n".join(lines) + ("\n" if lines else ""))
        return path

    def save_text(self, name: str, text: str) -> str:
        path = self.path(name, ".txt")
        self._write(path, text if text.endswith("\n") else text + "\n")
        return path

    def save_csv(
        self,
        name: str,
        headers: Sequence[str],
        rows: Iterable[Sequence[Any]],
    ) -> str:
        path = self.path(name, ".csv")
        out: List[str] = [",".join(str(h) for h in headers)]
        for row in rows:
            cells = []
            for value in row:
                text = "" if value is None else str(value)
                if "," in text:
                    text = f'"{text}"'
                cells.append(text)
            out.append(",".join(cells))
        self._write(path, "\n".join(out) + "\n")
        return path

    def save_tracker(self, name: str = "metrics") -> Dict[str, str]:
        """Persist the accumulated metric tracker as JSON + JSONL + TXT."""
        paths = {
            "json": self.save_json(name, self.tracker.to_dict()),
            "jsonl": self.save_jsonl(name, self.tracker.to_records()),
        }
        headers, rows = self.tracker.to_rows()
        if rows:
            paths["txt"] = self.save_text(name, format_value_table(headers, rows))
        return paths

    def save_summary(self, name: str, summary: Mapping[str, Any]) -> str:
        """Write a ``key: value`` summary text file and the matching JSON."""
        lines = [f"{self.experiment} summary", "=" * 40]
        for key, value in summary.items():
            if isinstance(value, float) and math.isfinite(value):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")
        text = "\n".join(lines)
        self.save_json(f"{name}_summary", dict(summary))
        return self.save_text(name, text)

    # -- internals ---------------------------------------------------------
    def _write(self, path: str, text: str) -> None:
        if not self.save_artifacts:
            self.logger.debug("Artifact saving disabled; would have written %s", path)
            return
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        self.artifacts[os.path.basename(path)] = path
        self.logger.info("wrote %s", path)

    @property
    def wall_time(self) -> float:
        """Seconds elapsed since the logger was created."""
        return time.time() - self.t0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": self.experiment,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "wall_time": self.wall_time,
            "artifacts": dict(self.artifacts),
            "num_records": len(self.tracker),
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks of the logging helpers (no GPU, no datasets)."""
    import tempfile

    report: Dict[str, Any] = {}
    logger = setup_logging("WARNING", name=f"{ROOT_LOGGER_NAME}._selftest", force=True)
    report["logger_name"] = logger.name
    report["level"] = logging.getLevelName(logger.level)

    tracker = MetricTracker(keys=["f1", "f2"], log_every=1, logger=logger)
    tracker.add(0, f1=3.21, f2=200)
    tracker.add(1, f1=2.10, f2=193)
    tracker.add(2, f1=1.92, f2=190.7)
    tracker.add(3, f1=float("nan"), f2=None)  # must be dropped
    report["curve_f1"] = tracker.curve("f1")
    report["curve_f2"] = tracker.curve("f2")
    report["num_records"] = len(tracker)
    report["best_f1"] = tracker.best("f1")
    report["last_f2"] = tracker.latest("f2")
    report["summary"] = tracker.summary()
    report["passes_drop_nonfinite"] = len(tracker) == 3 and len(tracker.curve("f1")) == 3

    headers, rows = tracker.to_rows()
    table = format_value_table(headers, rows)
    report["table_lines"] = len(table.splitlines())
    report["passes_table_shape"] = len(rows) == 3 and headers[:2] == ["iter", "f1"]

    jsonable = to_jsonable({"a": [1, 2], "b": (3, 4), "c": float("inf"), "d": None})
    report["passes_jsonable"] = jsonable["c"] is None and jsonable["b"] == [3, 4]

    cfg = LoggingConfig.from_dict({"level": "DEBUG", "log_every": 5, "bogus_key": 1})
    report["config"] = cfg.to_dict()
    report["passes_config"] = cfg.level == "DEBUG" and cfg.log_every == 5

    with tempfile.TemporaryDirectory() as tmp:
        elog = ExperimentLogger("selftest", output_dir=tmp, logger=logger)
        elog.log_metrics("k200_eps0.2", {"f1": 1.92, "f2": 190.7, "acc": 70.6})
        elog.log_metrics("k200_eps0.3", {"f1": 2.26, "f2": 185.0, "acc": 70.1})
        paths = elog.save_tracker("metrics")
        elog.save_json("payload", {"results": [1, 2, 3]})
        elog.save_text("table", format_value_table(["k", "f1"], [[200, 1.92]]))
        elog.save_summary("selftest", {"f1_mean": 1.92, "f2_mean": 190.7})
        report["artifacts"] = {k: os.path.basename(v) for k, v in paths.items()}
        report["passes_artifacts"] = all(os.path.exists(p) for p in paths.values())
        with open(paths["json"], "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        report["passes_roundtrip"] = loaded["curves"]["f1"] == [1.92, 2.26]

    report["ok"] = all(
        report[k]
        for k in (
            "passes_drop_nonfinite",
            "passes_table_shape",
            "passes_jsonable",
            "passes_config",
            "passes_artifacts",
            "passes_roundtrip",
        )
    )

    if verbose:
        print("logging self-test")
        for key, value in report.items():
            print(f"  {key}: {value}")
    return report


if __name__ == "__main__":  # pragma: no cover - manual invocation
    logging.basicConfig(level=logging.INFO)
    _selftest(verbose=True)
