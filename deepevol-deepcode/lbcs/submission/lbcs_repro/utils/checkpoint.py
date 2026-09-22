"""Checkpointing and results-persistence glue for the LBCS reproduction.

This module is *pure glue*: it contains no paper-specific formula. It provides

* model / optimizer / scheduler state persistence (``save_model_state``,
  ``load_model_state``, ``save_checkpoint``, ``load_checkpoint``,
  ``restore_model``),
* a small :class:`CheckpointManager` that keeps the best-*N* checkpoints of a
  run ordered by a monitored metric (mirroring what the experiment drivers need
  when they train an inner-loop proxy ``theta(m)`` or a post-selection target
  model),
* serialization of experiment result tables / metric traces to disk (JSON,
  JSONL, CSV, plain text) on top of :mod:`lbcs_repro.utils.logging`.

Everything degrades gracefully when PyTorch is unavailable: state-dict helpers
raise a clear :class:`ImportError`, while JSON/CSV/text helpers keep working so
offline (numpy-only) self-tests and unit checks still run.

Run the offline self-test with::

    python -m lbcs_repro.utils.checkpoint
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # optional soft dependency
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a hard dependency in practice
    _np = None  # type: ignore

try:  # pragma: no cover - exercised only when torch is present
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


LOGGER = logging.getLogger(__name__)

__all__ = [
    "CheckpointManager",
    "CheckpointInfo",
    "checkpoint_state",
    "list_checkpoints",
    "load_checkpoint",
    "load_model_state",
    "restore_model",
    "save_checkpoint",
    "save_model_state",
    "save_results",
    "save_json",
    "load_json",
    "save_jsonl",
    "load_jsonl",
    "save_csv",
    "save_text",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_RESULTS_DIR",
]

DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_RESULTS_DIR = "results"


# --------------------------------------------------------------------------- #
# torch helpers
# --------------------------------------------------------------------------- #
def torch_available() -> bool:
    """Return ``True`` when PyTorch could be imported."""
    return _TORCH_AVAILABLE


def _require_torch(what: str) -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            f"{what} requires PyTorch, which could not be imported. "
            "Install torch/torchvision to use model checkpointing."
        )


def _unwrap(model: Any) -> Any:
    """Return the underlying module for ``nn.DataParallel``/``DistributedDataParallel``."""
    if hasattr(model, "module") and not hasattr(model, "state_dict"):
        return model.module
    if hasattr(model, "module"):
        # DataParallel exposes state_dict itself, but storing the bare module is
        # more portable across devices. Only unwrap when the attribute really is
        # a module.
        inner = getattr(model, "module")
        if _TORCH_AVAILABLE and isinstance(inner, torch.nn.Module):
            return inner
    return model


def _as_cpu(state: Any) -> Any:
    """Recursively move tensors in a state-dict-like structure to CPU."""
    if _TORCH_AVAILABLE and isinstance(state, torch.Tensor):
        return state.detach().cpu()
    if isinstance(state, dict):
        return {k: _as_cpu(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(_as_cpu(v) for v in state)
    return state


# --------------------------------------------------------------------------- #
# paths / directories
# --------------------------------------------------------------------------- #
def checkpoint_state() -> Dict[str, Any]:
    """Describe the global checkpoint/RNG state in a JSON-friendly mapping.

    Used by the experiment drivers to record *which* random state produced a
    result, satisfying the plan's reproducibility requirement that seeds are
    logged for every repeat. Contains only ``json``-serializable primitives.
    """
    info: Dict[str, Any] = {
        "timestamp": time.time(),
        "pid": os.getpid(),
        "torch_available": _TORCH_AVAILABLE,
    }
    try:
        import random

        info["python_random_state"] = list(random.getstate()[1])
    except Exception:  # pragma: no cover - defensive
        pass
    if _np is not None:
        try:
            info["numpy_random_state"] = [
                int(x) for x in _np.random.get_state()[1].tolist()
            ]
            info["numpy_random_pos"] = int(_np.random.get_state()[2])
        except Exception:  # pragma: no cover - defensive
            pass
    if _TORCH_AVAILABLE:
        try:
            info["torch_random_state"] = torch.get_rng_state().tolist()
        except Exception:  # pragma: no cover
            pass
        try:
            if torch.cuda.is_available():
                info["cuda_rng_state_0"] = torch.cuda.get_rng_state(0).tolist()
        except Exception:  # pragma: no cover
            pass
    return info


def _ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def _join(directory: Optional[str], name: str) -> str:
    if not directory:
        return name
    return os.path.join(directory, name)


# --------------------------------------------------------------------------- #
# model / optimizer state
# --------------------------------------------------------------------------- #
def save_model_state(
    model: Any,
    path: str,
    *,
    optimizer: Any = None,
    scheduler: Any = None,
    epoch: Optional[int] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
    to_cpu: bool = True,
    include_rng: bool = False,
    atomic: bool = True,
) -> str:
    """Persist model (and optionally optimizer/scheduler) state to ``path``.

    Parameters
    ----------
    model:
        Anything exposing ``state_dict`` (a ``torch.nn.Module`` or a plain
        mapping, which is then stored verbatim).
    path:
        Destination file. Parent directories are created automatically.
    optimizer, scheduler:
        Optional objects exposing ``state_dict``.
    epoch, metrics, extra, config:
        Arbitrary JSON-serializable bookkeeping stored alongside the weights.
    to_cpu:
        Move tensors to CPU before saving (portable across devices).
    include_rng:
        Also store the global RNG state (see :func:`checkpoint_state`).

    Returns
    -------
    str
        The path written.
    """
    _require_torch("save_model_state")

    if isinstance(model, Mapping):
        model_state: Any = dict(model)
    else:
        module = _unwrap(model)
        if not hasattr(module, "state_dict"):
            raise TypeError("model must expose a `state_dict()` method")
        model_state = module.state_dict()

    payload: Dict[str, Any] = {
        "format": "lbcs-checkpoint",
        "version": 1,
        "model": _as_cpu(model_state) if to_cpu else model_state,
        "saved_at": time.time(),
    }
    if optimizer is not None:
        try:
            payload["optimizer"] = _as_cpu(optimizer.state_dict()) if to_cpu else optimizer.state_dict()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Could not serialize optimizer state: %s", exc)
    if scheduler is not None:
        try:
            payload["scheduler"] = _as_cpu(scheduler.state_dict()) if to_cpu else scheduler.state_dict()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Could not serialize scheduler state: %s", exc)
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if metrics is not None:
        payload["metrics"] = dict(metrics)
    if config is not None:
        payload["config"] = dict(config)
    if extra:
        payload["extra"] = dict(extra)
    if include_rng:
        payload["rng"] = checkpoint_state()

    directory = os.path.dirname(os.path.abspath(path))
    _ensure_dir(directory)
    if atomic:
        tmp = f"{path}.tmp"
        torch.save(payload, tmp)
        shutil.move(tmp, path)
    else:
        torch.save(payload, path)
    LOGGER.debug("Saved model state to %s", path)
    return path


def load_model_state(
    model: Any,
    path: str,
    *,
    optimizer: Any = None,
    scheduler: Any = None,
    map_location: Optional[str] = None,
    strict: bool = True,
    return_metadata: bool = False,
) -> Any:
    """Load weights stored by :func:`save_model_state` into ``model``.

    Returns the :class:`torch.nn.Module` (or ``(model, metadata)`` when
    ``return_metadata=True``), where ``metadata`` contains the bookkeeping keys
    (``epoch``, ``metrics``, ``config``, ``extra``, ``rng``).
    """
    _require_torch("load_model_state")

    if map_location is None:
        map_location = "cpu"
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(payload, Mapping) and "model" in payload and "format" in payload:
        model_state = payload["model"]
        metadata: Dict[str, Any] = {
            k: v for k, v in payload.items() if k not in ("model", "optimizer", "scheduler", "format", "version")
        }
        if optimizer is not None and "optimizer" in payload:
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to restore optimizer state: %s", exc)
        if scheduler is not None and "scheduler" in payload:
            try:
                scheduler.load_state_dict(payload["scheduler"])
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to restore scheduler state: %s", exc)
    else:  # bare state dict
        model_state = payload
        metadata = {}

    target = _unwrap(model)
    if isinstance(target, Mapping):
        raise TypeError(
            "load_model_state expects a torch.nn.Module; use load_json for plain mappings"
        )
    missing, unexpected = target.load_state_dict(model_state, strict=strict)
    if missing or unexpected:  # pragma: no cover - only with strict=False
        LOGGER.warning("State dict mismatch - missing=%s unexpected=%s", missing, unexpected)
    LOGGER.debug("Loaded model state from %s", path)
    if return_metadata:
        return model, metadata
    return model


def restore_model(
    model: Any,
    path: str,
    *,
    optimizer: Any = None,
    scheduler: Any = None,
    map_location: Optional[str] = None,
    strict: bool = True,
) -> Any:
    """Alias of :func:`load_model_state` with training-friendly semantics."""
    return load_model_state(
        model,
        path,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location=map_location,
        strict=strict,
    )


# --------------------------------------------------------------------------- #
# generic checkpoints (arbitrary payload)
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: str,
    *,
    model: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    epoch: Optional[int] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    state: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
    include_rng: bool = False,
    atomic: bool = True,
) -> str:
    """Save a full training checkpoint (model/optimizer/scheduler/state).

    If ``model`` is given this delegates to :func:`save_model_state`. Otherwise
    the remaining keyword arguments form the payload, which is written with
    ``torch.save`` when available and falls back to JSON otherwise.
    """
    if model is not None:
        return save_model_state(
            model,
            path,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
            extra=extra,
            config=config,
            include_rng=include_rng,
            atomic=atomic,
        )

    payload: Dict[str, Any] = {
        "format": "lbcs-checkpoint",
        "version": 1,
        "saved_at": time.time(),
    }
    if state is not None:
        payload["state"] = dict(state)
    if optimizer is not None:
        payload["optimizer"] = _as_cpu(optimizer.state_dict())
    if scheduler is not None:
        payload["scheduler"] = _as_cpu(scheduler.state_dict())
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if metrics is not None:
        payload["metrics"] = dict(metrics)
    if config is not None:
        payload["config"] = dict(config)
    if extra:
        payload["extra"] = dict(extra)
    if include_rng:
        payload["rng"] = checkpoint_state()

    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    if _TORCH_AVAILABLE:
        if atomic:
            tmp = f"{path}.tmp"
            torch.save(payload, tmp)
            shutil.move(tmp, path)
        else:
            torch.save(payload, path)
    else:  # JSON fallback so glue code is testable without torch
        save_json(path, _jsonify(payload), indent=2)
    LOGGER.debug("Saved checkpoint to %s", path)
    return path


def load_checkpoint(
    path: str,
    *,
    map_location: Optional[str] = None,
    load_model_into: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Returns the raw payload as a dict. When ``load_model_into`` is supplied the
    model/optimizer/scheduler states are restored into the provided objects.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    payload: Any
    if _TORCH_AVAILABLE:
        try:
            payload = torch.load(
                path, map_location=map_location or "cpu", weights_only=False
            )
        except Exception:
            payload = _load_json_or_none(path)
    else:
        loaded = _load_json_or_none(path)
        if loaded is None:
            raise ImportError(
                "load_checkpoint needs PyTorch to read this file (or a JSON checkpoint)."
            )
        payload = loaded

    if not isinstance(payload, Mapping):
        raise ValueError(f"Unrecognized checkpoint payload in {path!r}")

    result = dict(payload)
    if load_model_into is not None and "model" in result:
        restored = _unwrap(load_model_into)
        restored.load_state_dict(result["model"], strict=strict)
        if optimizer is not None and "optimizer" in result:
            try:
                optimizer.load_state_dict(result["optimizer"])
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to restore optimizer state: %s", exc)
        if scheduler is not None and "scheduler" in result:
            try:
                scheduler.load_state_dict(result["scheduler"])
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to restore scheduler state: %s", exc)
    return result


# --------------------------------------------------------------------------- #
# listing checkpoints
# --------------------------------------------------------------------------- #
def list_checkpoints(
    directory: str = DEFAULT_CHECKPOINT_DIR,
    *,
    pattern: str = ".pt",
    sort_by: str = "mtime",
    descending: bool = True,
) -> List[str]:
    """List checkpoint files inside ``directory``.

    ``pattern`` may be a substring (``".pt"``) or a glob (``"*.pt"``). Results
    are sorted by ``"mtime"`` (default), ``"name"`` or ``"size"``.
    """
    if not directory or not os.path.isdir(directory):
        return []

    files: List[str] = []
    for entry in os.listdir(directory):
        full = os.path.join(directory, entry)
        if not os.path.isfile(full):
            continue
        if not pattern:
            files.append(full)
        elif any(ch in pattern for ch in "*?["):
            import fnmatch

            if fnmatch.fnmatch(entry, pattern):
                files.append(full)
        elif pattern in entry:
            files.append(full)

    if sort_by == "name":
        files.sort(key=os.path.basename, reverse=descending)
    elif sort_by == "size":
        files.sort(key=lambda p: os.path.getsize(p), reverse=descending)
    else:
        files.sort(key=lambda p: os.path.getmtime(p), reverse=descending)
    return files


# --------------------------------------------------------------------------- #
# results persistence
# --------------------------------------------------------------------------- #
def save_json(path: str, payload: Any, indent: int = 2) -> str:
    """Write ``payload`` as JSON (creating parent directories)."""
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonify(payload), fh, indent=indent, sort_keys=False)
    LOGGER.debug("Wrote JSON to %s", path)
    return path


def load_json(path: str, default: Any = None) -> Any:
    """Read JSON from ``path``; return ``default`` when missing/unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        LOGGER.debug("Could not read JSON %s: %s", path, exc)
        return default


def save_jsonl(path: str, records: Iterable[Any]) -> str:
    """Write one JSON object per line (metric traces, per-repeat records)."""
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(_jsonify(record)))
            fh.write("\n")
            count += 1
    LOGGER.debug("Wrote %d JSONL records to %s", count, path)
    return path


def load_jsonl(path: str) -> List[Any]:
    """Read a JSONL file into a list (empty list when missing)."""
    out: List[Any] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    except (OSError, ValueError) as exc:
        LOGGER.debug("Could not read JSONL %s: %s", path, exc)
    return out


def save_csv(path: str, headers: Sequence[Any], rows: Iterable[Sequence[Any]]) -> str:
    """Write a CSV table with the given ``headers`` and ``rows``."""
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(headers))
        for row in rows:
            writer.writerow(list(row))
    LOGGER.debug("Wrote CSV to %s", path)
    return path


def save_text(path: str, text: str) -> str:
    """Write plain text (rendered paper-style tables)."""
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    LOGGER.debug("Wrote text to %s", path)
    return path


def save_results(
    results: Any,
    output_dir: str = DEFAULT_RESULTS_DIR,
    *,
    name: str = "results",
    tables: Optional[Mapping[str, str]] = None,
    config: Optional[Mapping[str, Any]] = None,
    records: Optional[Iterable[Any]] = None,
    summary: Optional[Mapping[str, Any]] = None,
    save_json_artifact: bool = True,
    save_csv_artifact: bool = False,
    csv_headers: Optional[Sequence[Any]] = None,
    csv_rows: Optional[Iterable[Sequence[Any]]] = None,
) -> Dict[str, str]:
    """Persist an experiment's result bundle and return the written paths.

    The bundle mirrors what every driver returns: a (JSON-friendly) results
    object, optionally rendered table strings, raw per-repeat ``records``, a
    ``summary`` mapping, and optional CSV cells. Files are written as
    ``<output_dir>/<name>.json``, ``.jsonl``, ``.csv``, ``.txt``.

    Returns
    -------
    dict
        ``{"json": path, "jsonl": path, "csv": path, "txt": path}`` with missing
        entries omitted.
    """
    _ensure_dir(output_dir)
    written: Dict[str, str] = {}

    if save_json_artifact:
        payload: Dict[str, Any] = {"results": results}
        if config is not None:
            payload["config"] = config
        if summary is not None:
            payload["summary"] = summary
        written["json"] = save_json(_join(output_dir, f"{name}.json"), payload)
    if records is not None:
        written["jsonl"] = save_jsonl(_join(output_dir, f"{name}_raw.jsonl"), records)
    if tables:
        blob = "\n\n".join(f"{title}\n{text}" for title, text in tables.items())
        written["txt"] = save_text(_join(output_dir, f"{name}.txt"), blob)
    if save_csv_artifact or csv_rows is not None:
        headers = csv_headers if csv_headers is not None else []
        rows = csv_rows if csv_rows is not None else []
        written["csv"] = save_csv(_join(output_dir, f"{name}.csv"), headers, rows)
    LOGGER.info("Saved results to %s (%s)", output_dir, ", ".join(sorted(written)))
    return written


# --------------------------------------------------------------------------- #
# checkpoint manager
# --------------------------------------------------------------------------- #
@dataclass
class CheckpointInfo:
    """Bookkeeping entry for one saved checkpoint."""

    path: str
    epoch: Optional[int] = None
    metric: Optional[float] = None
    tag: str = ""
    saved_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "epoch": self.epoch,
            "metric": self.metric,
            "tag": self.tag,
            "saved_at": self.saved_at,
        }


@dataclass
class CheckpointManager:
    """Keep the best-``max_to_keep`` checkpoints ordered by a monitored metric.

    Parameters
    ----------
    output_dir:
        Directory receiving the checkpoint files.
    monitor:
        Name of the metric used for ranking (default ``"accuracy"``).
    mode:
        ``"max"`` (higher is better, default) or ``"min"`` (``f1``/loss).
    max_to_keep:
        Number of checkpoints retained; the rest are deleted.
    prefix:
        File-name prefix (usually the experiment name).
    save_optimizer:
        Persist optimizer/scheduler state alongside the weights.
    save_best_only:
        When ``True`` only strictly better checkpoints are written.
    """

    output_dir: str = DEFAULT_CHECKPOINT_DIR
    monitor: str = "accuracy"
    mode: str = "max"
    max_to_keep: int = 2
    prefix: str = "model"
    save_optimizer: bool = True
    save_best_only: bool = False
    logger: Optional[logging.Logger] = None
    history: List[CheckpointInfo] = field(default_factory=list)
    _counter: int = field(default=0, init=False, repr=False)
    _best: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.mode = str(self.mode).lower()
        if self.mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        if self.logger is None:
            self.logger = LOGGER
        _ensure_dir(self.output_dir)

    # -- ranking helpers ---------------------------------------------------- #
    def _better(self, value: float, reference: float) -> bool:
        return value > reference if self.mode == "max" else value < reference

    def is_improvement(self, value: Optional[float]) -> bool:
        """Return ``True`` when ``value`` improves on the best seen so far."""
        if value is None:
            return self._best is None
        if self._best is None:
            return True
        return self._better(float(value), float(self._best))

    @property
    def best(self) -> Optional[float]:
        """Best monitored metric value seen so far."""
        return self._best

    # -- main API ----------------------------------------------------------- #
    def save(
        self,
        model: Any = None,
        *,
        metric: Optional[float] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        optimizer: Any = None,
        scheduler: Any = None,
        tag: str = "",
        state: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
        force: bool = False,
    ) -> Optional[str]:
        """Save a checkpoint and prune the archive.

        The monitored value defaults to ``metric``, falling back to
        ``metrics[self.monitor]``. Returns the written path, or ``None`` when
        the checkpoint was skipped (``save_best_only`` and no improvement).
        """
        value = metric
        if value is None and metrics is not None:
            value = metrics.get(self.monitor)  # type: ignore[union-attr]

        improved = self.is_improvement(value)
        if self.save_best_only and not improved and not force:
            self.logger.debug(
                "Skipping checkpoint: %s=%.4f did not improve best=%s",
                self.monitor,
                float(value) if value is not None else float("nan"),
                self._best,
            )
            return None

        self._counter += 1
        epoch_part = f"epoch{epoch:04d}" if epoch is not None else f"step{self._counter:04d}"
        tag_part = f"_{tag}" if tag else ""
        suffix = "best" if improved else "last"
        filename = f"{self.prefix}_{epoch_part}_{suffix}{tag_part}.pt"
        path = _join(self.output_dir, filename)

        if model is not None:
            save_model_state(
                model,
                path,
                optimizer=optimizer if self.save_optimizer else None,
                scheduler=scheduler if self.save_optimizer else None,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )
        else:
            save_checkpoint(
                path,
                epoch=epoch,
                metrics=metrics,
                state=state,
                config=config,
            )

        info = CheckpointInfo(
            path=path,
            epoch=epoch,
            metric=None if value is None else float(value),
            tag=tag,
            saved_at=time.time(),
        )
        self.history.append(info)
        if improved and value is not None:
            self._best = float(value)
        self.prune()
        return path

    # convenience alias used by training loops
    step = save

    def prune(self) -> List[str]:
        """Delete checkpoints beyond ``max_to_keep`` (worst first)."""
        if self.max_to_keep is None or self.max_to_keep <= 0:
            return []
        # Entries without a metric are treated as the worst ones.
        def key(item: CheckpointInfo) -> Tuple[int, float]:
            missing = 1 if item.metric is None else 0
            value = 0.0 if item.metric is None else item.metric
            signed = -value if self.mode == "max" else value
            return (missing, signed)

        ordered = sorted(self.history, key=key)
        keep = set(id(info) for info in ordered[: self.max_to_keep])
        removed: List[str] = []
        retained: List[CheckpointInfo] = []
        for info in self.history:
            if id(info) in keep:
                retained.append(info)
                continue
            try:
                if os.path.exists(info.path):
                    os.remove(info.path)
                    removed.append(info.path)
            except OSError as exc:  # pragma: no cover - defensive
                self.logger.warning("Could not remove checkpoint %s: %s", info.path, exc)
        self.history = retained
        return removed

    # -- introspection ------------------------------------------------------ #
    def best_checkpoint(self) -> Optional[CheckpointInfo]:
        """Return the checkpoint with the best monitored metric."""
        items = [info for info in self.history if info.metric is not None]
        if not items:
            return None
        return (max if self.mode == "max" else min)(items, key=lambda i: i.metric)  # type: ignore[arg-type]

    def paths(self, sort_by: str = "metric") -> List[str]:
        """List retained checkpoint paths, best first by default."""
        infos = list(self.history)
        if sort_by == "metric":
            infos.sort(
                key=lambda i: (
                    1 if i.metric is None else 0,
                    (i.metric if self.mode == "min" else -(i.metric or 0.0)),
                )
            )
        elif sort_by == "epoch":
            infos.sort(key=lambda i: (i.epoch is None, i.epoch or 0))
        return [info.path for info in infos]

    def load_best(
        self,
        model: Any,
        *,
        map_location: Optional[str] = None,
        optimizer: Any = None,
        scheduler: Any = None,
        strict: bool = True,
    ) -> Any:
        """Restore the best retained checkpoint into ``model``."""
        best = self.best_checkpoint()
        if best is None:
            raise FileNotFoundError("No checkpoints with a monitored metric were saved")
        return load_model_state(
            model,
            best.path,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
            strict=strict,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "monitor": self.monitor,
            "mode": self.mode,
            "max_to_keep": self.max_to_keep,
            "prefix": self.prefix,
            "best": self._best,
            "checkpoints": [info.to_dict() for info in self.history],
        }


# --------------------------------------------------------------------------- #
# JSON coercion helpers
# --------------------------------------------------------------------------- #
def _jsonify(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-serializable primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if _np is not None:
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    if isinstance(obj, Mapping):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]
    if _TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "to_dict"):
        try:
            return _jsonify(obj.to_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    return str(obj)


def _load_json_or_none(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline (torch-optional) validation of the checkpoint glue."""
    import tempfile

    report: Dict[str, Any] = {"ok": True, "torch_available": _TORCH_AVAILABLE}
    with tempfile.TemporaryDirectory() as tmp:
        # --- results persistence -------------------------------------------
        paths = save_results(
            {"f1": 1.92, "f2": 190.7},
            os.path.join(tmp, "res"),
            name="table1",
            tables={"Table 1": "k | eps | f1"},
            records=[{"k": 200, "f1": 1.92}, {"k": 400, "f1": 1.05}],
            config={"epsilon": 0.2},
            summary={"repeats": 20},
            save_csv_artifact=True,
            csv_headers=["k", "f1"],
            csv_rows=[[200, 1.92], [400, 1.05]],
        )
        report["artifacts"] = sorted(paths)
        assert {"json", "jsonl", "csv", "txt"} <= set(paths), paths
        loaded = load_json(paths["json"])
        assert loaded["results"]["f1"] == 1.92
        assert len(load_jsonl(paths["jsonl"])) == 2

        # --- checkpoint state ----------------------------------------------
        state = checkpoint_state()
        assert "timestamp" in state and "torch_available" in state

        # --- manager bookkeeping -------------------------------------------
        mgr = CheckpointManager(
            output_dir=os.path.join(tmp, "ckpt"),
            monitor="accuracy",
            mode="max",
            max_to_keep=2,
            prefix="unit",
        )
        accuracies = [70.0, 75.0, 72.0, 80.0]
        for epoch, acc in enumerate(accuracies):
            mgr.save(None, metric=acc, epoch=epoch, state={"epoch": epoch})
        report["best"] = mgr.best
        assert mgr.best == 80.0, mgr.best
        assert len(mgr.history) == 2, [i.to_dict() for i in mgr.history]
        kept = mgr.paths()
        assert len(kept) == 2
        assert "step" not in kept[0]  # sanity: files exist
        best_info = mgr.best_checkpoint()
        assert best_info is not None and best_info.metric == 80.0
        report["checkpoints"] = [info.to_dict() for info in mgr.history]

        # min-mode manager
        mgr_min = CheckpointManager(
            output_dir=os.path.join(tmp, "ckpt_min"),
            monitor="f1",
            mode="min",
            max_to_keep=1,
            prefix="unit",
        )
        for epoch, f1 in enumerate([3.2, 2.4, 1.9]):
            mgr_min.save(None, metric=f1, epoch=epoch)
        assert mgr_min.best == 1.9, mgr_min.best
        assert len(mgr_min.history) == 1

        # --- listing --------------------------------------------------------
        listed = list_checkpoints(os.path.join(tmp, "ckpt"), pattern=".pt")
        assert listed, "expected saved checkpoint files"
        report["num_listed"] = len(listed)

        # --- torch state round trip ----------------------------------------
        if _TORCH_AVAILABLE:
            import torch.nn as nn

            model = nn.Linear(4, 2)
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            path = os.path.join(tmp, "linear.pt")
            before = [p.detach().clone() for p in model.parameters()]
            save_model_state(model, path, optimizer=opt, epoch=3, metrics={"accuracy": 88.0})
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(1.0)
            model, meta = load_model_state(model, path, optimizer=opt, return_metadata=True)
            after = [p.detach() for p in model.parameters()]
            assert all(torch.allclose(a, b) for a, b in zip(before, after))
            assert meta.get("epoch") == 3
            assert meta.get("metrics", {}).get("accuracy") == 88.0
            report["torch_roundtrip"] = True

    if verbose:
        print("checkpoint self-test:")
        print(f"  torch available : {report['torch_available']}")
        print(f"  artifacts       : {report.get('artifacts')}")
        print(f"  checkpoints     : {len(report.get('checkpoints', []))}")
        print(f"  all checks      : {'OK' if report['ok'] else 'FAILED'}")
    return report


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _selftest(verbose=True)
