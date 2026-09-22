"""Logging and provenance helpers for the Robust CLIP reproduction package.

This module contains *no* paper-specific formula.  It is pure glue used by every
entry point (``main.py``, ``eval_imagenet.py``, ``eval_vqa.py``,
``eval_captioning.py``, ``eval_jailbreak.py``, ``train_robust_clip.py``) to:

* configure a consistent ``logging`` setup (console + optional file handler),
* log hyper-parameter *provenance* so that values the paper/Addendum never state
  are reported as externally supplied instead of silently invented,
* pretty-print configs / result dictionaries (JSON-safe conversion), and
* keep the raw-vs-normalized pixel-space bookkeeping visible in run logs.

The provenance sentinels (``UNSPECIFIED_BY_ADDENDUM`` / ``EXTERNAL_DEFAULT``)
mirror the constants defined in :mod:`robust_clip_repro`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

__all__ = [
    "UNSPECIFIED",
    "EXTERNAL_DEFAULT",
    "PAPER_BODY",
    "ADDENDUM",
    "UPSTREAM",
    "PROVENANCE_TAGS",
    "LOGGER_NAME",
    "get_logger",
    "setup_logging",
    "log_config",
    "log_provenance",
    "classify_provenance",
    "split_config_by_provenance",
    "ProvenanceReport",
    "jsonable",
    "to_json",
    "save_json",
    "load_json",
    "log_metric",
    "log_table",
    "format_table",
    "Timer",
    "log_pixel_space",
    "attach_file_handler",
    "quiet_logging",
    "configure_from_args",
    "provenance_metadata",
    "collect_provenance",
    "print_json",
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_DATE_FORMAT",
]

# ---------------------------------------------------------------------------
# Provenance sentinels (mirror robust_clip_repro.__init__ to avoid a hard
# import cycle; the values are identical on purpose).
# ---------------------------------------------------------------------------
UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"
EXTERNAL_DEFAULT = "EXTERNAL_DEFAULT"
PAPER_BODY = "PAPER_BODY"
ADDENDUM = "ADDENDUM"
UPSTREAM = "UPSTREAM"

PROVENANCE_TAGS: Tuple[str, ...] = (
    PAPER_BODY,
    ADDENDUM,
    UPSTREAM,
    EXTERNAL_DEFAULT,
    UNSPECIFIED,
)

LOGGER_NAME = "robust_clip_repro"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a package logger (``robust_clip_repro`` by default)."""
    if not name:
        return logging.getLogger(LOGGER_NAME)
    if name == LOGGER_NAME or name.startswith(LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def setup_logging(
    verbose: bool = True,
    *,
    level: Optional[int] = None,
    log_file: Optional[Union[str, os.PathLike]] = None,
    fmt: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
    force: bool = False,
) -> logging.Logger:
    """Configure the package logger with a console handler and optional file.

    ``verbose=False`` switches the console handler to ``WARNING`` (useful for
    library-style embedding); ``level`` overrides the console level explicitly.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if level is None:
        level = logging.DEBUG if verbose else logging.INFO
    console_level = logging.INFO if verbose else logging.WARNING
    if level is not None and not verbose:
        console_level = max(console_level, logging.WARNING)

    logger.setLevel(min(level, console_level))
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover - defensive
                pass

    if not any(getattr(h, "_rcr_console", False) for h in logger.handlers):
        stream = logging.StreamHandler(stream=sys.stdout)
        stream.setLevel(console_level)
        stream.setFormatter(formatter)
        stream._rcr_console = True  # type: ignore[attr-defined]
        logger.addHandler(stream)
    else:
        for handler in logger.handlers:
            if getattr(handler, "_rcr_console", False):
                handler.setLevel(console_level)

    if log_file is not None:
        attach_file_handler(log_file, fmt=fmt, datefmt=datefmt, level=level)

    logger.propagate = False
    return logger


def attach_file_handler(
    log_file: Union[str, os.PathLike],
    *,
    fmt: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
    level: int = logging.DEBUG,
    logger: Optional[logging.Logger] = None,
) -> logging.Handler:
    """Attach a file handler to the package logger (idempotent per path)."""
    logger = logger or logging.getLogger(LOGGER_NAME)
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == resolved:
            return handler
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    handler._rcr_file = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return handler


class quiet_logging:
    """Context manager that temporarily raises the package logger level."""

    def __init__(self, level: int = logging.ERROR, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(LOGGER_NAME)
        self.level = level
        self._previous: Optional[int] = None

    def __enter__(self) -> "quiet_logging":
        self._previous = self.logger.level
        self.logger.setLevel(self.level)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._previous is not None:
            self.logger.setLevel(self._previous)


def configure_from_args(args: Any, *, log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging from an ``argparse.Namespace``-like object."""
    verbose = True
    if args is not None:
        verbose = bool(getattr(args, "verbose", True))
    if log_file is None and args is not None:
        log_file = getattr(args, "log_file", None)
    return setup_logging(verbose=verbose, log_file=log_file)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def jsonable(obj: Any, *, _depth: int = 0) -> Any:
    """Recursively convert ``obj`` into something ``json.dumps`` can handle."""
    if _depth > 12:  # pragma: no cover - defensive against pathological nesting
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v, _depth=_depth + 1) for k, v in obj.items()}
    if is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(asdict(obj), _depth=_depth + 1)
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v, _depth=_depth + 1) for v in obj]
    # torch tensors / numpy arrays / objects with tolist()
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return jsonable(tolist(), _depth=_depth + 1)
        except Exception:  # pragma: no cover - defensive
            pass
    shape = getattr(obj, "shape", None)
    if shape is not None:
        return f"<array shape={tuple(shape)}>"
    as_dict = getattr(obj, "as_dict", None)
    if callable(as_dict):
        try:
            return jsonable(as_dict(), _depth=_depth + 1)
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(obj, "__dict__"):
        return {str(k): jsonable(v, _depth=_depth + 1) for k, v in vars(obj).items()}
    return str(obj)


def to_json(obj: Any, *, indent: int = 2) -> str:
    """Serialize ``obj`` to a deterministic (sorted-key) JSON string."""
    try:
        return json.dumps(jsonable(obj), indent=indent, sort_keys=True)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return json.dumps(str(obj))


def save_json(obj: Any, path: Union[str, os.PathLike], *, indent: int = 2) -> str:
    """Write ``obj`` as JSON to ``path``; returns the resolved path string."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(obj, indent=indent), encoding="utf-8")
    return str(path)


def load_json(path: Union[str, os.PathLike]) -> Any:
    """Load a JSON file (returns ``None`` when the file does not exist)."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def print_json(payload: Any, *, stream: Optional[Any] = None) -> None:
    """Print ``payload`` as JSON (used by the CLI mode dispatchers)."""
    stream = stream or sys.stdout
    print(to_json(payload), file=stream)


# ---------------------------------------------------------------------------
# Provenance bookkeeping
# ---------------------------------------------------------------------------
def classify_provenance(key: str, value: Any = None, provenance: Optional[Mapping[str, Any]] = None) -> str:
    """Classify one config ``key`` into a provenance tag.

    Explicit ``provenance`` mappings (as written in ``configs/*.yaml`` under a
    ``provenance:`` block) win.  Otherwise a small keyword heuristic is applied,
    and anything genuinely unknown is reported as ``UNSPECIFIED`` rather than
    being attributed to the paper.
    """
    key_l = str(key).lower()

    if isinstance(provenance, Mapping):
        for tag, keys in provenance.items():
            tag_name = str(tag).upper()
            if tag_name not in PROVENANCE_TAGS:
                # tolerate lowercase / spaced variants, e.g. "paper body"
                tag_name = tag_name.replace(" ", "_")
            if isinstance(keys, (list, tuple, set)):
                normalized = {str(k).lower() for k in keys}
            elif isinstance(keys, Mapping):
                normalized = {str(k).lower() for k in keys.keys()}
            elif isinstance(keys, str):
                normalized = {keys.lower()}
            else:
                continue
            if key_l in normalized:
                return tag_name if tag_name in PROVENANCE_TAGS else UNSPECIFIED

    addendum_keys = {
        "momentum",
        "random_start",
        "uniform_init",
        "gradient_normalization",
        "gradient_norm_eps",
        "projection_space",
        "precision",
        "precision_half",
        "precision_single",
        "int_dtype_half",
        "int_dtype_single",
        "quant_scale",
        "iterations",
        "alpha",
        "targeted",
        "single_source_image",
        "source_image",
        "target_corpus",
        "eval_prompts",
        "top_k",
        "num_ground_truths",
        "warm_start_single",
        "threshold",
        "trust_remote_code",
        "clip_model",
        "clip_pretrained",
        "clip_resolution",
        "use_openclip",
        "patch_hf_clip",
        "skip_word_attack",
        "vqa_target_maybe",
        "vqa_target_word",
    }
    paper_keys = {
        "dataset_name",
        "attacks",
        "apgd_iterations",
        "eps",
        "epsilons",
        "num_robust_samples",
        "resolution",
        "image_size",
        "norm",
        "loss",
        "restarts",
    }
    upstream_keys = {"rho", "eot_iter", "n_target_classes", "alpha_factor", "init", "optimizer", "micro_batch_targets"}

    if key_l in addendum_keys:
        return ADDENDUM
    if key_l in paper_keys:
        return PAPER_BODY
    if key_l in upstream_keys:
        return UPSTREAM
    return UNSPECIFIED


@dataclass
class ProvenanceReport:
    """Structured view of which config values come from where."""

    addendum: List[str] = field(default_factory=list)
    paper_body: List[str] = field(default_factory=list)
    upstream: List[str] = field(default_factory=list)
    external_default: List[str] = field(default_factory=list)
    unspecified: List[str] = field(default_factory=list)

    def add(self, tag: str, key: str) -> None:
        bucket = {
            ADDENDUM: self.addendum,
            PAPER_BODY: self.paper_body,
            UPSTREAM: self.upstream,
            EXTERNAL_DEFAULT: self.external_default,
            UNSPECIFIED: self.unspecified,
        }.get(tag, self.unspecified)
        if key not in bucket:
            bucket.append(key)

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            "addendum": sorted(self.addendum),
            "paper_body": sorted(self.paper_body),
            "upstream": sorted(self.upstream),
            "external_default": sorted(self.external_default),
            "unspecified_by_addendum": sorted(self.unspecified),
        }

    def unspecified_keys(self) -> List[str]:
        return sorted(set(self.unspecified) | set(self.external_default))

    def __bool__(self) -> bool:
        return bool(self.addendum or self.paper_body or self.upstream or self.external_default or self.unspecified)


def split_config_by_provenance(
    cfg: Mapping[str, Any],
    provenance: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], ProvenanceReport]:
    """Flatten ``cfg`` (one level, excluding nested ``provenance``) and tag every key."""
    provenance = provenance
    if provenance is None and isinstance(cfg, Mapping):
        provenance = cfg.get("provenance")  # type: ignore[assignment]

    flat: Dict[str, Any] = {}
    report = ProvenanceReport()
    for key, value in (cfg or {}).items():
        if str(key).lower() == "provenance":
            continue
        if isinstance(value, Mapping):
            # Flatten nested blocks (attack:, model:, precision:, ...) one level.
            for sub_key, sub_value in value.items():
                if str(sub_key).lower() == "provenance":
                    continue
                if isinstance(sub_value, Mapping):
                    continue  # keep depth shallow; log as-is
                flat[str(sub_key)] = sub_value
                report.add(classify_provenance(sub_key, sub_value, provenance), str(sub_key))
            continue
        flat[str(key)] = value
        report.add(classify_provenance(key, value, provenance), str(key))
    return flat, report


def log_provenance(
    cfg: Mapping[str, Any],
    *,
    provenance: Optional[Mapping[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    title: Optional[str] = None,
    max_unspecified: int = 40,
) -> ProvenanceReport:
    """Log which config values are paper-stated vs. externally supplied.

    Values the Addendum does not state are explicitly *not* attributed to the
    paper: they are reported as externally supplied / unspecified.
    """
    logger = logger or get_logger()
    _, report = split_config_by_provenance(cfg, provenance)

    if title:
        logger.info("%s", title)
    if report.addendum:
        logger.info("ADDENDUM-stated values: %s", ", ".join(sorted(report.addendum)))
    if report.paper_body:
        logger.info("paper-body values: %s", ", ".join(sorted(report.paper_body)))
    if report.upstream:
        logger.info("upstream defaults: %s", ", ".join(sorted(report.upstream)))

    external = report.unspecified_keys()
    if external:
        shown = external[:max_unspecified]
        suffix = "" if len(external) <= max_unspecified else f" (+{len(external) - max_unspecified} more)"
        logger.warning(
            "externally supplied / UNSPECIFIED_BY_ADDENDUM values (not from the paper): %s%s",
            ", ".join(shown),
            suffix,
        )
    return report


def collect_provenance(
    cfg: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    *,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a JSON-friendly provenance metadata record for a run."""
    cfg = cfg or {}
    if provenance is None and isinstance(cfg, Mapping):
        provenance = cfg.get("provenance")  # type: ignore[assignment]
    _, report = split_config_by_provenance(cfg, provenance)
    payload: Dict[str, Any] = {
        "unspecified_marker": UNSPECIFIED,
        "external_default_marker": EXTERNAL_DEFAULT,
        "provenance": report.as_dict(),
        "unspecified_keys": report.unspecified_keys(),
    }
    if extra:
        payload["extra"] = jsonable(extra)
    return payload


def provenance_metadata(
    cfg: Optional[Mapping[str, Any]] = None,
    *,
    external_defaults: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Public helper: provenance report plus explicitly declared external defaults."""
    payload = collect_provenance(cfg, extra=extra)
    if external_defaults:
        payload["external_defaults"] = jsonable(external_defaults)
        payload["external_default_keys"] = sorted(str(k) for k in external_defaults.keys())
    return payload


# ---------------------------------------------------------------------------
# Config / metric logging
# ---------------------------------------------------------------------------
def log_config(
    cfg: Any,
    *,
    logger: Optional[logging.Logger] = None,
    title: str = "configuration",
    provenance: Optional[Mapping[str, Any]] = None,
    log_provenance_report: bool = True,
) -> ProvenanceReport:
    """Log a config mapping/dataclass (JSON pretty-printed) plus provenance."""
    logger = logger or get_logger()
    as_mapping = cfg
    if is_dataclass(cfg) and not isinstance(cfg, type):
        as_mapping = asdict(cfg)
    if not isinstance(as_mapping, Mapping):
        as_mapping = {"value": as_mapping}

    logger.info("%s:\n%s", title, to_json(as_mapping))
    if log_provenance_report:
        return log_provenance(as_mapping, provenance=provenance, logger=logger, title=f"{title} provenance")
    return ProvenanceReport()


def log_metric(
    name: str,
    value: Any,
    *,
    logger: Optional[logging.Logger] = None,
    step: Optional[int] = None,
    precision: int = 4,
) -> None:
    """Log a single scalar metric."""
    logger = logger or get_logger()
    if isinstance(value, float):
        rendered: Any = f"{value:.{precision}f}"
    else:
        rendered = value
    if step is None:
        logger.info("%s = %s", name, rendered)
    else:
        logger.info("%s = %s (step %s)", name, rendered, step)


def _cell_width(values: Sequence[str]) -> int:
    return max((len(v) for v in values), default=0)


def format_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Optional[Sequence[str]] = None,
    percent: bool = True,
    float_precision: int = 2,
) -> str:
    """Render a list of row mappings as a plain-text table."""
    rows = list(rows or [])
    if not rows:
        return "(empty table)"

    if columns is None:
        columns = []
        for row in rows:
            for key in row.keys():
                if key not in columns:
                    columns.append(key)
    columns = list(columns)

    rendered: List[List[str]] = []
    for row in rows:
        line: List[str] = []
        for col in columns:
            value = row.get(col, "")
            if value is None:
                line.append("-")
            elif isinstance(value, float):
                if percent and 0.0 <= value <= 1.0:
                    line.append(f"{value * 100.0:.{float_precision}f}")
                else:
                    line.append(f"{value:.{float_precision}f}")
            else:
                line.append(str(value))
        rendered.append(line)

    widths = [_cell_width([col] + [r[i] for r in rendered]) for i, col in enumerate(columns)]
    header = " | ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
    sep = "-+-".join("-" * widths[i] for i in range(len(columns)))
    body = "\n".join(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)) for line in rendered)
    return f"{header}\n{sep}\n{body}"


def log_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    logger: Optional[logging.Logger] = None,
    title: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
    percent: bool = True,
) -> str:
    """Format and log a table; returns the rendered text."""
    logger = logger or get_logger()
    text = format_table(rows, columns=columns, percent=percent)
    if title:
        logger.info("%s\n%s", title, text)
    else:
        logger.info("\n%s", text)
    return text


class Timer:
    """Simple wall-clock timer used by the evaluators."""

    def __init__(self, name: str = "run", *, logger: Optional[logging.Logger] = None) -> None:
        self.name = name
        self.logger = logger or get_logger()
        self.start: float = time.time()
        self.elapsed: float = 0.0

    def reset(self) -> None:
        self.start = time.time()
        self.elapsed = 0.0

    def stop(self, *, log: bool = True) -> float:
        self.elapsed = time.time() - self.start
        if log:
            self.logger.info("%s finished in %.2fs", self.name, self.elapsed)
        return self.elapsed

    def __enter__(self) -> "Timer":
        self.reset()
        self.logger.info("%s started", self.name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop(log=exc_type is None)


def log_pixel_space(
    raw_pixels: Any,
    *,
    normalized_pixels: Any = None,
    logger: Optional[logging.Logger] = None,
    label: str = "pixels",
) -> Dict[str, Any]:
    """Log the raw-vs-normalized bookkeeping (Addendum: l_inf ball in raw space).

    Never mutates tensors; only reads ``min``/``max``/``shape``/``dtype``.
    """
    logger = logger or get_logger()

    def _describe(tensor: Any) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        shape = getattr(tensor, "shape", None)
        if shape is not None:
            info["shape"] = tuple(int(s) for s in shape)
        dtype = getattr(tensor, "dtype", None)
        if dtype is not None:
            info["dtype"] = str(dtype)
        for fn_name in ("min", "max"):
            fn = getattr(tensor, fn_name, None)
            if callable(fn):
                try:
                    value = fn()
                    value = getattr(value, "item", lambda: value)()
                    info[fn_name] = float(value)
                except Exception:  # pragma: no cover - defensive
                    pass
        return info

    payload = {"raw": _describe(raw_pixels)}
    if normalized_pixels is not None:
        payload["normalized"] = _describe(normalized_pixels)
    logger.info(
        "%s space: raw=%s%s | attack ball is computed around NON-normalized pixels",
        label,
        payload["raw"],
        f" | normalized={payload['normalized']}" if normalized_pixels is not None else "",
    )
    return payload


# ---------------------------------------------------------------------------
# Self-test / CLI
# ---------------------------------------------------------------------------
def _self_test(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks for the logging/provenance helpers (no torch needed)."""
    results: Dict[str, Any] = {}

    logger = setup_logging(verbose=verbose, force=True)
    results["logger_name"] = logger.name

    cfg = {
        "dataset_name": "ImageNet",
        "eps": None,
        "alpha": None,
        "iterations": 50,
        "momentum": 0.9,
        "random_start": True,
        "projection_space": "pixel",
        "precision": "single",
        "batch_size": 1,
        "seed": 0,
        "provenance": {
            "addendum": ["momentum", "random_start", "projection_space", "precision"],
            "paper_body": ["eps", "epsilons"],
            "unspecified_by_addendum": ["alpha", "iterations", "batch_size", "seed"],
        },
    }

    flat, report = split_config_by_provenance(cfg)
    report_d = report.as_dict()
    results["report"] = report_d
    assert "momentum" in report_d["addendum"], report_d
    assert "eps" in report_d["paper_body"], report_d
    assert set(["alpha", "iterations", "batch_size", "seed"]).issubset(set(report_d["unspecified_by_addendum"])), report_d
    assert "provenance" not in flat

    # Unspecified must never be attributed to the paper.
    tag = classify_provenance("some_new_knob", 3)
    assert tag == UNSPECIFIED, tag
    results["unknown_tag"] = tag

    # JSON round-trip (dataclass, nested mapping, tuple, Path-like).
    payload = {
        "cfg": cfg,
        "rows": [{"eps": "2/255", "clean": 0.75, "robust": 0.4}],
        "tuple": (1, 2),
    }
    text = to_json(payload)
    assert json.loads(text)["rows"][0]["clean"] == 0.75

    # Table rendering.
    table = format_table([{"name": "ViT-L-14", "clean": 0.75, "robust": 0.41}])
    assert "ViT-L-14" in table and "75.00" in table, table
    results["table"] = table.splitlines()[0]

    # Provenance metadata helper mentions both markers.
    meta = provenance_metadata(cfg, external_defaults={"pgd_iterations": None})
    assert meta["unspecified_marker"] == UNSPECIFIED
    assert meta["external_default_marker"] == EXTERNAL_DEFAULT
    assert "pgd_iterations" in meta["external_default_keys"]
    results["metadata"] = meta

    # Timer.
    with Timer("self-test", logger=logger):
        time.sleep(0.0)
    results["ok"] = True
    if verbose:
        logger.info("logging self-test passed")
    return results


def build_arg_parser() -> "argparse.ArgumentParser":  # noqa: F821 - lazy import below
    import argparse

    parser = argparse.ArgumentParser(description="Robust CLIP logging/provenance utilities")
    parser.add_argument("--self-test", action="store_true", help="run offline self-tests and exit")
    parser.add_argument("--config", type=str, default=None, help="optional YAML/JSON config to summarize")
    parser.add_argument("--json", action="store_true", help="print the self-test result as JSON")
    parser.add_argument("--quiet", action="store_true", help="suppress info-level logging")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=not args.quiet, force=True)
    logger = get_logger()

    if args.config:
        cfg: Any = None
        path = Path(args.config)
        if path.suffix.lower() in (".json",):
            cfg = load_json(path)
        else:
            try:
                import yaml  # type: ignore

                cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.error("could not read config %s: %s", path, exc)
                return 2
        log_config(cfg or {}, logger=logger, title=f"config: {path}")

    if args.self_test or not args.config:
        result = _self_test(verbose=not args.quiet)
        if args.json:
            print_json(result)
        else:
            logger.info("self-test: %s", "PASS" if result.get("ok") else "FAIL")
        return 0 if result.get("ok") else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
