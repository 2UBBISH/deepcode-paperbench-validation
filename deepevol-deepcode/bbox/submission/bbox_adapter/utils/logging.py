"""Logging, run-directory and checkpoint-naming utilities for BBox-Adapter.

This module is glue code: the paper (Section ``utils`` of the reproduction plan) does
not specify a logging design, so the choices here are convenience defaults that keep
every experiment entry point observable and reproducible.

Design goals
------------
1.  One call (``setup_logging``) configures console + optional file logging and
    returns a configured :class:`logging.Logger`.
2.  ``RunLogger`` owns the run directory layout used by the training scripts::

        <output_dir>/
            logs/train.log
            metrics.jsonl          # append-only scalar stream (one JSON per step)
            config.json            # resolved config snapshot
            seed.json              # describe_seed_state() output
            checkpoints/           # adapter_step_*.pt / adapter_final.pt
            curves/<name>.csv      # Appendix-K style energy / loss curves

3.  ``MetricTracker`` accumulates scalars (loss, mean positive energy, mean negative
    energy, accuracy ...) with running means so the Appendix-K figures and the
    Fig. 3(b) score curve can be plotted straight from ``metrics.jsonl``.
4.  ``checkpoint_name`` / ``resolve_checkpoint`` implement the checkpoint naming
    convention used by ``scripts/run_experiment.py`` and
    ``scripts/run_plug_and_play.py`` (plug-and-play loads a checkpoint trained for a
    *different* black-box model, hence the ``<dataset>_<size>_<blackbox>`` stem).

Everything is dependency-free except an optional NumPy/Matplotlib use in
``write_curve_csv`` (CSV only, no plotting) so the module imports on a bare Python.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "LOG_FORMAT",
    "DATE_FORMAT",
    "LEVELS",
    "DEFAULT_LOG_LEVEL",
    "ColoredFormatter",
    "MetricTracker",
    "RunLogger",
    "setup_logging",
    "get_logger",
    "log_config",
    "json_safe",
    "to_jsonable",
    "write_json",
    "write_jsonl",
    "read_jsonl",
    "write_csv",
    "write_curve_csv",
    "read_curve_csv",
    "checkpoint_name",
    "checkpoint_path",
    "resolve_checkpoint",
    "list_checkpoints",
    "latest_checkpoint",
    "count_checkpoints",
    "format_seconds",
    "Timer",
    "progress",
    "kv_table",
    "table",
    "make_run_dir",
    "describe_environment",
    "append_metrics",
    "load_metrics",
    "mean_of",
    "summary_stats",
]

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LEVELS: Tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
DEFAULT_LOG_LEVEL = "INFO"

_LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[41;37m",
}
_COLOR_RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class ColoredFormatter(logging.Formatter):
    """Console formatter that colorizes the level name when attached to a TTY."""

    def __init__(self, fmt: str = LOG_FORMAT, datefmt: str = DATE_FORMAT,
                 use_color: Optional[bool] = None) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        if use_color is None:
            use_color = _supports_color()
        self.use_color = bool(use_color)

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        text = super().format(record)
        if not self.use_color:
            return text
        color = _LEVEL_COLORS.get(record.levelname)
        if not color:
            return text
        return text.replace(record.levelname, f"{color}{record.levelname}{_COLOR_RESET}", 1)


def _supports_color(stream: Any = None) -> bool:
    """Best-effort TTY/ANSI detection; honours the ``NO_COLOR`` convention."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream if stream is not None else sys.stdout
    try:
        return bool(getattr(stream, "isatty", lambda: False)())
    except Exception:  # pragma: no cover - defensive
        return False


# ---------------------------------------------------------------------------
# Logger construction
# ---------------------------------------------------------------------------


def setup_logging(
    name: str = "bbox_adapter",
    *,
    level: str = DEFAULT_LOG_LEVEL,
    log_file: Optional[str] = None,
    stream: bool = True,
    use_color: Optional[bool] = None,
    fmt: str = LOG_FORMAT,
    datefmt: str = DATE_FORMAT,
    reset: bool = True,
) -> logging.Logger:
    """Configure and return a logger for one run.

    Parameters
    ----------
    name:
        Logger name (use the module/experiment name so records are filterable).
    level:
        One of :data:`LEVELS` (case-insensitive) or an ``int`` logging level.
    log_file:
        Optional path; parent directories are created automatically.
    stream:
        Whether to also log to stderr/stdout.
    reset:
        Remove previously installed handlers so repeated calls (e.g. two runs in a
        notebook) do not duplicate lines.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_coerce_level(level))
    logger.propagate = False

    if reset:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover - defensive
                pass

    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(_coerce_level(level))
        stream_handler.setFormatter(ColoredFormatter(fmt, datefmt, use_color=use_color))
        logger.addHandler(stream_handler)

    if log_file:
        path = os.fspath(log_file)
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setLevel(_coerce_level(level))
        # Plain formatter for files: no ANSI escapes in artifacts.
        file_handler.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(file_handler)

    if not logger.handlers:  # pragma: no cover - only when stream=False and no file
        logger.addHandler(logging.NullHandler())
    return logger


def get_logger(name: str = "bbox_adapter", *, level: Optional[str] = None) -> logging.Logger:
    """Return an existing logger, creating a console-only one if necessary."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name, level=level or DEFAULT_LOG_LEVEL)
    if level is not None:
        logger.setLevel(_coerce_level(level))
    return logger


def _coerce_level(level: Any) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        value = logging.getLevelName(level.strip().upper())
        if isinstance(value, int):
            return value
    return logging.INFO


def log_config(logger: logging.Logger, config: Any, *, title: str = "Configuration",
               level: int = logging.INFO) -> None:
    """Pretty-log a config mapping / dataclass (used to snapshot Appendix-H settings)."""
    payload = to_jsonable(config)
    logger.log(level, "%s:\n%s", title, json.dumps(payload, indent=2, sort_keys=True, default=str))


# ---------------------------------------------------------------------------
# JSON / CSV serialization helpers
# ---------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    """Recursively convert a value into something ``json.dumps`` accepts."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return None  # NaN / inf are not valid JSON
        return value
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):  # numpy / torch tensors
        try:
            return json_safe(value.tolist())
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(value, "item"):  # 0-dim tensors / numpy scalars
        try:
            return json_safe(value.item())
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(value, "to_dict"):
        try:
            return json_safe(value.to_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    return str(value)


# Plan/codebase friendly alias.
to_jsonable = json_safe


def write_json(path: str, payload: Any, *, indent: int = 2) -> str:
    """Write ``payload`` as JSON, creating parent directories."""
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=indent, sort_keys=False)
        handle.write("\n")
    return path


def write_jsonl(path: str, records: Iterable[Any], *, append: bool = False) -> str:
    """Append (or write) an iterable of JSON-serializable records."""
    _ensure_parent(path)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_safe(record), sort_keys=False) + "\n")
    return path


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts (missing file -> empty list)."""
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # pragma: no cover - tolerate partial writes
                continue
    return out


def write_csv(path: str, rows: Sequence[Mapping[str, Any]],
              fieldnames: Optional[Sequence[str]] = None) -> str:
    """Write rows of scalars to CSV; header is inferred from the first row."""
    _ensure_parent(path)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_cell(row.get(k)) for k in fieldnames})
    return path


def _csv_cell(value: Any) -> Any:
    value = json_safe(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def write_curve_csv(path: str, curve: Mapping[str, Sequence[float]], *,
                    x_name: str = "step", x: Optional[Sequence[float]] = None) -> str:
    """Write a multi-series curve (Appendix-K energy/loss figures, Fig. 3(b)).

    ``curve`` maps series name -> sequence of values; all series must share a length.
    """
    names = list(curve.keys())
    lengths = {len(curve[n]) for n in names}
    if len(lengths) > 1:
        raise ValueError(f"curve series have mismatched lengths: {sorted(lengths)}")
    n = lengths.pop() if lengths else 0
    xs = list(x) if x is not None else list(range(n))
    if len(xs) != n:
        raise ValueError(f"x has length {len(xs)} but curve has {n} points")
    rows = [{x_name: xs[i], **{name: curve[name][i] for name in names}} for i in range(n)]
    return write_csv(path, rows, fieldnames=[x_name] + names)


def read_curve_csv(path: str) -> Dict[str, List[float]]:
    """Inverse of :func:`write_curve_csv`; numeric cells are cast back to float."""
    if not os.path.exists(path):
        return {}
    out: Dict[str, List[float]] = {}
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                try:
                    out.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    out.setdefault(key, []).append(value)  # type: ignore[arg-type]
    return out


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(os.fspath(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)


# ---------------------------------------------------------------------------
# Metric tracking
# ---------------------------------------------------------------------------


class MetricTracker:
    """Running mean/count/last-value tracker for scalar training metrics.

    Mirrors the quantities the paper plots in Appendix K (NCE loss, mean positive
    energy, mean negative energy) and Fig. 3(b) (accuracy vs outer iteration ``T``).
    """

    def __init__(self, names: Optional[Sequence[str]] = None, *, window: int = 0) -> None:
        self.names: List[str] = list(names or [])
        self.window = int(window)
        self._sum: Dict[str, float] = {}
        self._count: Dict[str, int] = {}
        self._last: Dict[str, float] = {}
        self._history: Dict[str, List[float]] = {}
        for name in self.names:
            self._init_name(name)

    def _init_name(self, name: str) -> None:
        self._sum.setdefault(name, 0.0)
        self._count.setdefault(name, 0)
        self._history.setdefault(name, [])
        self._last.setdefault(name, float("nan"))

    def update(self, values: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Dict[str, float]:
        """Add one observation per key; returns the current running means."""
        merged: Dict[str, Any] = {}
        if values:
            merged.update(values)
        merged.update(kwargs)
        for name, value in merged.items():
            number = _as_float(value)
            if number is None:
                continue
            self._init_name(name)
            if name not in self.names:
                self.names.append(name)
            self._sum[name] += number
            self._count[name] += 1
            self._last[name] = number
            self._history[name].append(number)
        return self.means()

    def means(self) -> Dict[str, float]:
        return {name: self.mean(name) for name in self.names}

    def mean(self, name: str) -> float:
        count = self._count.get(name, 0)
        if not count:
            return float("nan")
        return self._sum[name] / count

    def last(self, name: str) -> float:
        return self._last.get(name, float("nan"))

    def count(self, name: str) -> int:
        return self._count.get(name, 0)

    def history(self, name: str) -> List[float]:
        return list(self._history.get(name, []))

    def windowed_mean(self, name: str) -> float:
        """Mean over the last ``self.window`` observations (all if ``window <= 0``)."""
        values = self._history.get(name, [])
        if not values:
            return float("nan")
        if self.window > 0:
            values = values[-self.window:]
        return sum(values) / len(values)

    def reset(self, name: Optional[str] = None) -> None:
        names = [name] if name else list(self.names)
        for key in names:
            self._sum[key] = 0.0
            self._count[key] = 0
            self._history[key] = []
            self._last[key] = float("nan")

    def as_dict(self) -> Dict[str, float]:
        return self.means()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "names": list(self.names),
            "sum": dict(self._sum),
            "count": dict(self._count),
            "last": dict(self._last),
            "history": {k: list(v) for k, v in self._history.items()},
            "window": self.window,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.names = list(state.get("names", []))
        self._sum = {k: float(v) for k, v in dict(state.get("sum", {})).items()}
        self._count = {k: int(v) for k, v in dict(state.get("count", {})).items()}
        self._last = {k: float(v) for k, v in dict(state.get("last", {})).items()}
        self._history = {k: [float(x) for x in v]
                         for k, v in dict(state.get("history", {})).items()}
        self.window = int(state.get("window", self.window or 0))

    def __contains__(self, name: object) -> bool:
        return name in self._count

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        parts = [f"{n}={self.mean(n):.4g}" for n in self.names]
        return "MetricTracker(" + ", ".join(parts) + ")"


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(value, "detach"):
        try:
            return float(value.detach().float().mean().item())
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def append_metrics(path: str, record: Mapping[str, Any]) -> str:
    """Append one scalar record to a ``metrics.jsonl`` file."""
    return write_jsonl(path, [record], append=True)


def load_metrics(path: str) -> List[Dict[str, Any]]:
    """Read back a ``metrics.jsonl`` file."""
    return read_jsonl(path)


def mean_of(records: Iterable[Mapping[str, Any]], key: str) -> float:
    values = [_as_float(r.get(key)) for r in records]
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def summary_stats(values: Sequence[float]) -> Dict[str, float]:
    """Mean/std/min/max/n of a sequence (std is the population std, matching numpy)."""
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {"n": n, "mean": mean, "std": var ** 0.5, "min": min(vals), "max": max(vals)}


# ---------------------------------------------------------------------------
# Run directory + RunLogger
# ---------------------------------------------------------------------------


@dataclass
class RunLogger:
    """Owns a run directory and provides logging + metric/curve/checkpoint helpers."""

    output_dir: str
    name: str = "bbox_adapter"
    dataset: Optional[str] = None
    size: Optional[str] = None
    blackbox: Optional[str] = None
    level: str = DEFAULT_LOG_LEVEL
    log_file: Optional[str] = None
    write_config: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = str(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.curve_dir, exist_ok=True)
        self.logger = setup_logging(
            self.name,
            level=self.level,
            log_file=self.log_file or self.default_log_file,
            use_color=None,
        )
        self.metrics = MetricTracker()
        self._t0 = time.time()
        if self.extra:
            for key, value in self.extra.items():
                setattr(self, key, value)

    # -- paths -------------------------------------------------------------
    @property
    def logs_dir(self) -> str:
        return os.path.join(self.output_dir, "logs")

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.output_dir, "checkpoints")

    @property
    def curve_dir(self) -> str:
        return os.path.join(self.output_dir, "curves")

    @property
    def default_log_file(self) -> str:
        return os.path.join(self.logs_dir, "train.log")

    @property
    def metrics_file(self) -> str:
        return os.path.join(self.output_dir, "metrics.jsonl")

    @property
    def config_file(self) -> str:
        return os.path.join(self.output_dir, "config.json")

    @property
    def seed_file(self) -> str:
        return os.path.join(self.output_dir, "seed.json")

    def path(self, *parts: str) -> str:
        return os.path.join(self.output_dir, *parts)

    # -- logging -----------------------------------------------------------
    def info(self, msg: str, *args: Any) -> None:
        self.logger.info(msg, *args)

    def debug(self, msg: str, *args: Any) -> None:
        self.logger.debug(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self.logger.warning(msg, *args)

    def error(self, msg: str, *args: Any) -> None:
        self.logger.error(msg, *args)

    def header(self, config: Optional[Any] = None, *, seed_state: Optional[Mapping[str, Any]] = None,
               components: Optional[Mapping[str, Any]] = None) -> None:
        """Emit the standard run header (env + config + seed state)."""
        env = describe_environment()
        self.info("=" * 78)
        self.info("BBox-Adapter run %r (dataset=%s, size=%s, blackbox=%s)",
                  self.name, self.dataset, self.size, self.blackbox)
        self.info("output_dir: %s", self.output_dir)
        self.info("python: %s | executable: %s", env.get("python_version"), env.get("executable"))
        self.info("torch: %s | cuda: %s (%s device(s))", env.get("torch_version"),
                  env.get("cuda_available"), env.get("cuda_device_count"))
        if components:
            for key, value in components.items():
                self.info("  %-22s %s", key, value)
        if seed_state:
            self.info("seed state: %s", json.dumps(json_safe(seed_state), sort_keys=True))
        if config is not None and self.write_config:
            write_json(self.config_file, config)
            self.info("wrote config snapshot -> %s", self.config_file)
        self.info("=" * 78)

    def log_config(self, config: Any, *, title: str = "Configuration") -> None:
        log_config(self.logger, config, title=title)

    def log_metrics(self, step: Optional[int] = None, *, prefix: str = "",
                    record: bool = True, **values: Any) -> Dict[str, float]:
        """Log and optionally persist scalar metrics; returns the running means."""
        means = self.metrics.update(values) if values else self.metrics.means()
        if values:
            entry: Dict[str, Any] = {}
            if step is not None:
                entry["step"] = int(step)
            entry.update({k: _as_float(v) for k, v in values.items()})
            entry.update({"elapsed": round(time.time() - self._t0, 3)})
            if record:
                append_metrics(self.metrics_file, entry)
            rendered = " ".join(f"{k}={_as_float(v):.4g}" for k, v in values.items()
                                if _as_float(v) is not None)
            self.info("%s%s", prefix, rendered)
        return means

    # -- artifacts ---------------------------------------------------------
    def save_config(self, config: Any) -> str:
        return write_json(self.config_file, config)

    def save_seed_state(self, seed_state: Mapping[str, Any]) -> str:
        return write_json(self.seed_file, seed_state)

    def save_curve(self, name: str, curve: Mapping[str, Sequence[float]], **kwargs: Any) -> str:
        return write_curve_csv(self.path("curves", f"{name}.csv"), curve, **kwargs)

    def checkpoint_path(self, tag: Any = "final") -> str:
        return os.path.join(self.checkpoint_dir, checkpoint_name(
            dataset=self.dataset, size=self.size, blackbox=self.blackbox, tag=tag))

    def save_checkpoint(self, obj: Any, tag: Any = "final", *, save_fn: Optional[Callable[..., Any]] = None,
                        **kwargs: Any) -> str:
        """Save a checkpoint using ``obj.save_pretrained`` or ``torch.save``."""
        path = self.checkpoint_path(tag)
        if save_fn is not None:
            save_fn(obj, path, **kwargs)
        elif hasattr(obj, "save_pretrained"):
            obj.save_pretrained(path)
        else:
            import torch  # local import keeps this module import-light

            torch.save(obj, path)
        self.info("saved checkpoint -> %s", path)
        return path

    def load_checkpoint(self, tag: Any = "final", *, loader: Optional[Callable[..., Any]] = None,
                        **kwargs: Any) -> Any:
        path = resolve_checkpoint(self.checkpoint_dir, tag=tag)
        if loader is not None:
            return loader(path, **kwargs)
        import torch  # local import keeps this module import-light

        return torch.load(path, **kwargs)

    def summary(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "elapsed_seconds": round(time.time() - self._t0, 3),
            "metrics": self.metrics.means(),
            "counts": {name: self.metrics.count(name) for name in self.metrics.names},
        }


def make_run_dir(base: str, *, name: Optional[str] = None, timestamp: bool = False,
                 exist_ok: bool = True) -> str:
    """Create (and return) a run directory, optionally namespaced by timestamp."""
    parts = [os.fspath(base)]
    if name:
        parts.append(str(name))
    if timestamp:
        parts.append(time.strftime("%Y%m%d-%H%M%S"))
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=exist_ok)
    return path


# ---------------------------------------------------------------------------
# Checkpoint naming
# ---------------------------------------------------------------------------


def checkpoint_name(*, dataset: Optional[str] = None, size: Optional[str] = None,
                    blackbox: Optional[str] = None, tag: Any = "final",
                    prefix: str = "adapter", ext: str = "") -> str:
    """Build the canonical checkpoint file/dir name.

    Example: ``adapter_strategyqa_0.1b_gpt-3.5-turbo_final``.

    ``plug-and-play`` (Table 3) loads a checkpoint trained against gpt-3.5-turbo and
    evaluated with davinci-002 / Mixtral, so the black-box tag is part of the identity
    but *not* of the requested tag.
    """
    parts = [prefix]
    for value in (dataset, size, blackbox):
        if value:
            parts.append(_slug(str(value)))
    if tag is not None and str(tag) != "":
        parts.append(_slug(str(tag)))
    stem = "_".join(parts)
    return f"{stem}{ext}"


def _slug(text: str) -> str:
    cleaned = str(text).strip().replace(os.sep, "-").replace(" ", "-")
    return cleaned or "unknown"


def checkpoint_path(output_dir: str, tag: Any = "final", *, dataset: Optional[str] = None,
                    size: Optional[str] = None, blackbox: Optional[str] = None,
                    prefix: str = "adapter", ext: str = "") -> str:
    """Full path for a checkpoint inside ``output_dir`` (convenience wrapper)."""
    subdir = output_dir if os.path.basename(os.path.normpath(output_dir)) == "checkpoints" \
        else os.path.join(output_dir, "checkpoints")
    return os.path.join(subdir, checkpoint_name(dataset=dataset, size=size, blackbox=blackbox,
                                                tag=tag, prefix=prefix, ext=ext))


_STEP_TAG_RE = None


def _tag_sort_key(name: str) -> Tuple[int, float, str]:
    """Sort checkpoints naturally: numeric tags first (by value), then lexical."""
    stem = os.path.splitext(name)[0]
    pieces = stem.split("_")
    for piece in reversed(pieces):
        if piece.isdigit():
            return (0, float(piece), name)
    return (1, 0.0, name)


def list_checkpoints(output_dir: str, *, pattern: Optional[str] = None) -> List[str]:
    """List checkpoint entries (files or directories) inside ``output_dir``."""
    directory = output_dir
    if os.path.isdir(os.path.join(output_dir, "checkpoints")):
        directory = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(directory):
        return []
    names = sorted(os.listdir(directory), key=_tag_sort_key)
    out = []
    for name in names:
        if pattern and pattern not in name:
            continue
        if name in ("logs", "curves"):
            continue
        out.append(os.path.join(directory, name))
    return out


def count_checkpoints(output_dir: str, *, pattern: Optional[str] = None) -> int:
    return len(list_checkpoints(output_dir, pattern=pattern))


def latest_checkpoint(output_dir: str, *, pattern: Optional[str] = None) -> Optional[str]:
    """Return the checkpoint with the highest numeric tag (falling back to lexical)."""
    entries = list_checkpoints(output_dir, pattern=pattern)
    if not entries:
        return None
    stem_keys = [(os.path.basename(e), e) for e in entries]
    # Prefer the largest numeric tag; non-numeric tags sort after numeric ones.
    numeric = [(float(v), e) for (n, e) in stem_keys
               for v in _numeric_tags(n)]
    if numeric:
        return max(numeric, key=lambda kv: kv[0])[1]
    return stem_keys[-1][1]


def _numeric_tags(name: str) -> List[str]:
    stem = os.path.splitext(name)[0]
    return [p for p in stem.split("_") if p.isdigit()]


def resolve_checkpoint(where: str, tag: Any = "final", *,
                       dataset: Optional[str] = None, size: Optional[str] = None,
                       blackbox: Optional[str] = None) -> str:
    """Resolve a user-supplied checkpoint reference.

    Accepts (a) an existing path, (b) a directory containing ``checkpoints/``, or
    (c) an output directory from which the ``tag`` (or the latest checkpoint) is
    selected. Raises ``FileNotFoundError`` when nothing matches.
    """
    where = os.fspath(where)
    if os.path.exists(where):
        return where
    candidate = checkpoint_path(where, tag, dataset=dataset, size=size, blackbox=blackbox)
    if os.path.exists(candidate):
        return candidate
    directory = os.path.join(where, "checkpoints")
    if os.path.isdir(directory):
        matches = [p for p in list_checkpoints(directory) if str(tag) in os.path.basename(p)]
        if matches:
            return matches[-1]
        latest = latest_checkpoint(directory)
        if latest:
            return latest
    raise FileNotFoundError(
        f"no checkpoint found at {where!r} (tried {candidate!r}); "
        f"available: {[os.path.basename(p) for p in list_checkpoints(where)]}")


# ---------------------------------------------------------------------------
# Misc formatting helpers
# ---------------------------------------------------------------------------


def format_seconds(seconds: float) -> str:
    """Human-readable duration (``1h 02m 03s``)."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


class Timer:
    """Small wall-clock timer / context manager with ``split`` lap support."""

    def __init__(self, name: str = "timer", *, logger: Optional[logging.Logger] = None,
                 verbose: bool = False) -> None:
        self.name = name
        self.logger = logger
        self.verbose = verbose
        self.start: Optional[float] = None
        self.elapsed: float = 0.0
        self.laps: List[Tuple[str, float]] = []

    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.start is not None:
            self.elapsed = time.time() - self.start
        if self.verbose and self.logger is not None:
            self.logger.info("%s finished in %s", self.name, format_seconds(self.elapsed))
        return False

    def lap(self, label: str = "") -> float:
        now = time.time()
        base = self.start if self.start is not None else now
        delta = now - base
        self.laps.append((label, delta))
        self.start = now
        return delta

    def __float__(self) -> float:
        if self.start is not None:
            return time.time() - self.start
        return self.elapsed


def progress(iterable: Iterable[Any], *, total: Optional[int] = None, desc: str = "",
             enabled: bool = True, logger: Optional[logging.Logger] = None,
             log_every: int = 0) -> Iterable[Any]:
    """Minimal tqdm-compatible iterator: uses tqdm when available, else logs sparsely."""
    if not enabled:
        yield from iterable
        return
    try:
        from tqdm import tqdm  # type: ignore

        yield from tqdm(iterable, total=total, desc=desc or None, leave=False)
        return
    except Exception:  # pragma: no cover - tqdm optional
        pass
    n = 0
    for item in iterable:
        if log_every and logger is not None and (n + 1) % log_every == 0:
            logger.info("%s %d/%s", desc or "progress", n + 1,
                        total if total is not None else "?")
        n += 1
        yield item


def kv_table(mapping: Mapping[str, Any], *, indent: int = 0, sort_keys: bool = False) -> str:
    """Render a key/value mapping as a fixed-width text block."""
    items = list(mapping.items())
    if sort_keys:
        items.sort(key=lambda kv: str(kv[0]))
    width = max((len(str(k)) for k, _ in items), default=0)
    pad = " " * indent
    return "\n".join(f"{pad}{str(k):<{width}} : {v}" for k, v in items)


def table(rows: Sequence[Mapping[str, Any]], *, fields: Optional[Sequence[str]] = None,
          float_fmt: str = "{:.4f}") -> str:
    """Render rows of scalars as an aligned text table (paper-table style)."""
    rows = [dict(r) for r in rows]
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    cells: List[List[str]] = []
    for row in rows:
        line = []
        for f in fields:
            value = row.get(f, "")
            if isinstance(value, float):
                value = float_fmt.format(value)
            line.append(str(value))
        cells.append(line)
    widths = [max(len(str(f)), *(len(c[i]) for c in cells)) if cells else len(str(f))
              for i, f in enumerate(fields)]
    header = "  ".join(str(f).ljust(w) for f, w in zip(fields, widths))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(cell.ljust(w) for cell, w in zip(line, widths)) for line in cells]
    return "\n".join([header, sep] + body)


def describe_environment() -> Dict[str, Any]:
    """Collect runtime/environment facts for the run header (CPU/GPU libs optional)."""
    info: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "executable": sys.executable,
        "platform": sys.platform,
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "torch_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "gpu_names": [],
        "transformers_version": None,
        "env_flags": {k: os.environ.get(k) for k in ("PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES")
                      if os.environ.get(k) is not None},
    }
    try:
        import torch  # type: ignore

        info["torch_version"] = getattr(torch, "__version__", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            try:
                info["gpu_names"] = [torch.cuda.get_device_name(i)
                                     for i in range(info["cuda_device_count"])]
            except Exception:  # pragma: no cover - defensive
                info["gpu_names"] = []
    except Exception:  # pragma: no cover - torch optional
        pass
    try:
        import transformers  # type: ignore

        info["transformers_version"] = getattr(transformers, "__version__", None)
    except Exception:  # pragma: no cover - transformers optional
        pass
    return info


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------


def _self_test(tmp_dir: Optional[str] = None) -> Dict[str, Any]:
    """Dependency-free smoke test of logging, metrics, curves and checkpoint naming."""
    import tempfile

    results: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:  # fresh dir per call
        root = tmp_dir or tmp
        run = RunLogger(output_dir=root, name="bbox_selftest", dataset="strategyqa",
                        size="0.1b", blackbox="gpt-3.5-turbo", level="INFO")
        results["dirs"] = {
            "logs": os.path.isdir(run.logs_dir),
            "checkpoints": os.path.isdir(run.checkpoint_dir),
            "curves": os.path.isdir(run.curve_dir),
        }

        run.header({"training": {"lr": 5e-6, "batch_size": 64}}, seed_state={"seed": 0})
        for step in range(1, 6):
            run.log_metrics(step=step, loss=1.0 / step, pos_energy=0.5 * step,
                            neg_energy=-0.25 * step)

        results["metrics_file"] = os.path.exists(run.metrics_file)
        records = load_metrics(run.metrics_file)
        results["n_records"] = len(records)
        results["mean_loss"] = round(mean_of(records, "loss"), 6)
        results["last_pos"] = run.metrics.last("pos_energy")
        results["windowed"] = round(run.metrics.windowed_mean("loss"), 6)

        run.save_curve("energy_curves", {
            "loss": [r["loss"] for r in records],
            "pos_energy": [r["pos_energy"] for r in records],
            "neg_energy": [r["neg_energy"] for r in records],
        }, x=[r["step"] for r in records])
        curve = read_curve_csv(run.path("curves", "energy_curves.csv"))
        results["curve_series"] = sorted(curve.keys())
        results["curve_len"] = len(curve.get("loss", []))

        name = checkpoint_name(dataset="strategyqa", size="0.1b",
                               blackbox="gpt-3.5-turbo", tag=6000)
        results["checkpoint_name"] = name
        os.makedirs(os.path.join(run.checkpoint_dir, name), exist_ok=True)
        os.makedirs(os.path.join(run.checkpoint_dir, checkpoint_name(
            dataset="strategyqa", size="0.1b", blackbox="gpt-3.5-turbo", tag="final")),
            exist_ok=True)
        results["n_checkpoints"] = count_checkpoints(run.checkpoint_dir)
        results["latest"] = os.path.basename(latest_checkpoint(run.checkpoint_dir) or "")

        created = resolve_checkpoint(run.output_dir, tag="final")
        results["resolve_ok"] = os.path.exists(created)

        results["table"] = table([{"dataset": "StrategyQA", "acc": 71.62},
                                  {"dataset": "GSM8K", "acc": 73.86}], fields=["dataset", "acc"])
        results["format_seconds"] = format_seconds(3725)
        stats = summary_stats([66.59, 67.51, 72.90])
        results["stats_ok"] = abs(stats["mean"] - 69.0) < 1.0
        results["env_keys"] = sorted(k for k in describe_environment()
                                     if k in ("python_version", "torch_version", "cuda_available"))
        results["json_safe_nan"] = json_safe(float("nan")) is None
        results["timer_type"] = type(Timer("t")).__name__
    return results


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import pprint

    pprint.pprint(_self_test())
