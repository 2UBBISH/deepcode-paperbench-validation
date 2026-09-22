"""Checkpoint management utilities for the RICE reproduction.

RICE (Cheng et al., ICML 2024, PMLR 235) runs a three-part pipeline:

1. **Pre-training (Stage 0)** -- a sub-optimal / bottlenecked DRL policy ``pi`` is
   trained per application (SB3 ``MlpPolicy`` PPO for the dense & sparse MuJoCo
   games, ``[128,128,128,128]`` for Selfish Mining, ``[64,64,64]`` for CAGE-2, the
   DI-engine default VAC for MetaDrive).  ``pi`` is frozen afterwards and is
   explained in Stage 1 and refined in Stage 2.
2. **Explanation (Stage 1, Algorithm 1)** -- a mask network ``~pi_theta(a_t^m|s_t)``
   is trained with vanilla PPO plus the blinding bonus ``alpha * a_t^m``; the mask's
   probability of "keep" ``P(a_t^m = 0 | s_t)`` is the step-level state importance
   used to build the mixed initial state distribution
   ``mu(s) = beta * d_rho^{pi_hat}(s) + (1 - beta) * rho(s)``.
3. **Refining (Stage 2, Algorithm 2)** -- ``pi`` is refined into ``pi'`` with PPO on
   ``mu(s)`` and an RND intrinsic reward ``lambda * ||f(s_{t+1}) - f_hat(s_{t+1})||^2``.

Every artifact of that pipeline needs a stable on-disk home so that the
experiment drivers (``experiments/exp1..exp5``) and the CLI scripts
(``scripts/train_target.py``, ``scripts/train_mask.py``, ``scripts/run_fidelity.py``,
``scripts/run_refine.py``, ``scripts/run_ablation.py``) can share state without
re-training.  This module provides that layer:

* a deterministic *path convention* per (env, kind, seed) that matches the names
  hard-coded in the scripts (``policies/<env>_ppo[_seedN].zip``,
  ``policies/<env>_mask[_seedN].pt``, ...);
* type-aware save/load dispatch so callers can hand over an SB3 model, a native
  ``ActorCritic``, a ``MaskNetwork``, an ``RNDModule``, a ``PPORefiner`` /
  ``MaskTrainer`` / ``StateMaskTrainer``, or a plain ``state_dict``;
* a JSON *registry* (``policies/registry.json``) indexing every checkpoint with
  its metadata (env, kind, seed, format, timesteps, wall time, scores) so ranks
  and provenance can be inspected without importing torch.

The module is deliberately dependency-tolerant: ``torch`` / Stable-Baselines3 /
the ``rice`` sub-packages are imported defensively, so ``--help`` and the
registry are usable on a bare install.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_REGISTRY_NAME",
    "KIND_TARGET",
    "KIND_MASK",
    "KIND_STATEMASK",
    "KIND_REFINED",
    "KIND_RND",
    "KIND_REFINER",
    "KIND_EXPLANATION",
    "KINDS",
    "STEMS",
    "EXTENSIONS",
    "CheckpointRecord",
    "CheckpointManager",
    "checkpoint_dir",
    "checkpoint_path",
    "build_manager",
    "save_checkpoint",
    "load_checkpoint",
    "resolve_checkpoint",
    "list_checkpoints",
    "index_checkpoints",
    "prune_checkpoints",
    "read_registry",
    "write_registry",
    "detect_kind",
    "describe_manager",
    "build_arg_parser",
    "main",
]

_LOGGER = logging.getLogger("rice.tools.checkpoint_manager")

# ---------------------------------------------------------------------------
# Defensive imports of the project's IO helpers
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised implicitly
    from rice.utils.io import PROJECT_ROOT, ensure_dir, load_json, save_json  # type: ignore

    _HAS_IO = True
except Exception:  # pragma: no cover - fallback for standalone use
    _HAS_IO = False
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    def ensure_dir(path: str) -> str:  # type: ignore[misc]
        if path:
            os.makedirs(path, exist_ok=True)
        return path

    def _local_json_default(obj: Any) -> Any:  # type: ignore[misc]
        for attr in ("tolist", "item"):
            fn = getattr(obj, attr, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    continue
        try:
            return str(obj)
        except Exception:
            return None

    def save_json(obj: Any, path: str, indent: int = 2) -> str:  # type: ignore[misc]
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=indent, default=_local_json_default)
        return path

    def load_json(path: str, default: Optional[Any] = None) -> Any:  # type: ignore[misc]
        if not os.path.isfile(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return default


# ---------------------------------------------------------------------------
# Defensive imports of the RICE modules whose objects we persist
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised implicitly
    from rice.models.policies import load_policy, normalize_env_key, save_policy  # type: ignore

    _HAS_POLICIES = True
except Exception:  # pragma: no cover
    _HAS_POLICIES = False
    load_policy = None  # type: ignore[assignment]
    save_policy = None  # type: ignore[assignment]

    def normalize_env_key(env_id: Any) -> str:  # type: ignore[misc]
        if env_id is None:
            return "default"
        key = str(env_id).strip().lower()
        key = key.rsplit("/", 1)[-1]
        if key.endswith(".yaml") or key.endswith(".yml"):
            key = key.rsplit(".", 1)[0]
        for prefix in ("sparse-", "sparse_", "sparse"):
            if key.startswith(prefix) and key != "sparse":
                key = key[len(prefix):].lstrip("-_") or key
                key = f"sparse_{key}"
                break
        for suffix in ("-v0", "-v1", "-v2", "-v3", "-v4", "-v5"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
        key = key.replace("-", "_")
        aliases = {
            "metadrive": "autodriving",
            "meta_drive": "autodriving",
            "macrodrive": "autodriving",
            "cage": "cage2",
            "cage_2": "cage2",
            "selfishmining": "selfish_mining",
        }
        return aliases.get(key, key or "default")


try:  # pragma: no cover - exercised implicitly
    from rice.explanation.mask_network import (  # type: ignore
        load_mask_network,
        save_mask_network,
    )

    _HAS_MASK = True
except Exception:  # pragma: no cover
    _HAS_MASK = False
    load_mask_network = None  # type: ignore[assignment]
    save_mask_network = None  # type: ignore[assignment]

try:  # pragma: no cover
    from rice.refining.rnd import RNDModule  # type: ignore

    _HAS_RND = True
except Exception:  # pragma: no cover
    _HAS_RND = False
    RNDModule = None  # type: ignore[assignment]

try:  # pragma: no cover
    from rice.refining.ppo_refine import PPORefiner  # type: ignore

    _HAS_REFINER = True
except Exception:  # pragma: no cover
    _HAS_REFINER = False
    PPORefiner = None  # type: ignore[assignment]

try:  # pragma: no cover
    from rice.explanation.mask_trainer import MaskTrainer  # type: ignore

    _HAS_MASK_TRAINER = True
except Exception:  # pragma: no cover
    _HAS_MASK_TRAINER = False
    MaskTrainer = None  # type: ignore[assignment]

try:  # pragma: no cover
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants / conventions
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT_DIR = "policies"
DEFAULT_REGISTRY_NAME = "registry.json"

KIND_TARGET = "target"          # frozen pre-trained pi (Stage 0)
KIND_MASK = "mask"              # Stage-1 mask network (~pi_theta)
KIND_EXPLANATION = "explanation"  # alias of KIND_MASK (explanation artifact)
KIND_STATEMASK = "statemask"    # StateMask-R baseline mask network
KIND_REFINED = "refined"        # Stage-2 refined policy pi'
KIND_RND = "rnd"                # RND target + predictor (f, f_hat)
KIND_REFINER = "refiner"        # PPO refiner / mask-trainer state (resume)

KINDS: Tuple[str, ...] = (
    KIND_TARGET,
    KIND_MASK,
    KIND_STATEMASK,
    KIND_REFINED,
    KIND_RND,
    KIND_REFINER,
)

#: filename stems per kind -- the ``<key>_ppo`` / ``<key>_mask`` forms match the
#: hard-coded names used by ``scripts/train_target.py`` and ``scripts/train_mask.py``.
STEMS: Dict[str, str] = {
    KIND_TARGET: "ppo",
    KIND_MASK: "mask",
    KIND_EXPLANATION: "mask",
    KIND_STATEMASK: "statemask",
    KIND_REFINED: "refined",
    KIND_RND: "rnd",
    KIND_REFINER: "refiner",
}

#: default file extension per kind (SB3 ``.zip`` for the target policy, torch
#: ``.pt`` for everything else).
EXTENSIONS: Dict[str, str] = {
    KIND_TARGET: ".zip",
    KIND_MASK: ".pt",
    KIND_EXPLANATION: ".pt",
    KIND_STATEMASK: ".pt",
    KIND_REFINED: ".pt",
    KIND_RND: ".pt",
    KIND_REFINER: ".pt",
}

_ALIASES: Dict[str, str] = {
    "pi": KIND_TARGET,
    "policy": KIND_TARGET,
    "pretrained": KIND_TARGET,
    "pre_trained": KIND_TARGET,
    "target_policy": KIND_TARGET,
    "mask_net": KIND_MASK,
    "mask_network": KIND_MASK,
    "explain": KIND_EXPLANATION,
    "ours": KIND_MASK,
    "rice": KIND_MASK,
    "statemask_r": KIND_STATEMASK,
    "sm": KIND_STATEMASK,
    "refine": KIND_REFINED,
    "refined_policy": KIND_REFINED,
    "student": KIND_REFINED,
    "rnd_module": KIND_RND,
    "intrinsic": KIND_RND,
    "trainer": KIND_REFINER,
    "ppo_refiner": KIND_REFINER,
    "resume": KIND_REFINER,
}


def normalize_kind(kind: Any) -> str:
    """Canonicalise a checkpoint kind string."""
    if kind is None:
        return KIND_TARGET
    key = str(kind).strip().lower().replace("-", "_").replace(" ", "_")
    if key in STEMS:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    # tolerate suffixes such as "checkpoint_target"
    for known in tuple(STEMS) + tuple(_ALIASES):
        if key.endswith(known):
            return _ALIASES.get(known, known)
    return key


def checkpoint_dir(root: Optional[str] = None) -> str:
    """Return (and create) the checkpoint root directory."""
    path = root or os.path.join(PROJECT_ROOT, DEFAULT_CHECKPOINT_DIR)
    return ensure_dir(path)


def checkpoint_path(
    env_id: Any,
    kind: str = KIND_TARGET,
    seed: Optional[int] = None,
    out_dir: Optional[str] = None,
    tag: Optional[str] = None,
    ext: Optional[str] = None,
    backend: Optional[str] = None,
) -> str:
    """Canonical checkpoint path for ``(env_id, kind, seed)``.

    Mirrors the names produced by the RICE scripts, e.g.::

        checkpoint_path("Hopper-v3")                  -> policies/hopper_ppo.zip
        checkpoint_path("hopper", "mask", seed=1)     -> policies/hopper_mask_seed1.pt
        checkpoint_path("cage2", "refined", tag="ours")-> policies/cage2_ours_refined.pt

    ``backend="torch"`` forces the torch ``.pt`` extension even for the target
    policy (used by the native-PyTorch PPO fallback in ``scripts/train_target.py``).
    """
    key = normalize_env_key(env_id)
    kind_n = normalize_kind(kind)
    root = checkpoint_dir(out_dir)
    stem_map = {k: v for k, v in STEMS.items()}
    stem = stem_map.get(kind_n, str(kind_n))
    parts: List[str] = [key]
    if tag:
        parts.append(str(tag).strip().replace(" ", "_"))
    parts.append(stem)
    if seed is not None:
        parts.append(f"seed{int(seed)}")
    name = "_".join(parts)

    suffix = ext
    if suffix is None:
        suffix = EXTENSIONS.get(kind_n, ".pt")
        if backend is not None and str(backend).lower() in ("torch", "native", "pytorch"):
            suffix = ".pt"
        elif backend is not None and str(backend).lower() in ("sb3", "stable_baselines3", "zip"):
            suffix = ".zip"
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix
    return os.path.join(root, f"{name}{suffix or ''}")


# ---------------------------------------------------------------------------
# Record / registry
# ---------------------------------------------------------------------------
@dataclass
class CheckpointRecord:
    """One entry in the checkpoint registry."""

    env_id: str
    kind: str
    path: str
    seed: Optional[int] = None
    fmt: str = "pt"
    backend: Optional[str] = None
    timesteps: Optional[int] = None
    score: Optional[float] = None
    created: float = field(default_factory=time.time)
    size_bytes: Optional[int] = None
    tag: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    root: Optional[str] = None

    # ------------------------------------------------------------------
    @property
    def abs_path(self) -> str:
        """Absolute path, resolving relative entries against ``root``."""
        if os.path.isabs(self.path):
            return self.path
        base = self.root or os.path.join(PROJECT_ROOT, DEFAULT_CHECKPOINT_DIR)
        return os.path.abspath(os.path.join(base, self.path))

    @property
    def exists(self) -> bool:
        return os.path.isfile(self.abs_path)

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    def refresh(self) -> "CheckpointRecord":
        """Update ``exists``-dependent bookkeeping fields."""
        try:
            self.size_bytes = os.path.getsize(self.abs_path) if self.exists else None
        except OSError:
            self.size_bytes = None
        return self

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "env_id": self.env_id,
            "kind": self.kind,
            "path": self.path,
            "seed": self.seed,
            "format": self.fmt,
            "backend": self.backend,
            "timesteps": self.timesteps,
            "score": self.score,
            "created": self.created,
            "size_bytes": self.size_bytes,
            "tag": self.tag,
            "metadata": dict(self.metadata or {}),
            "exists": self.exists,
        }
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any], root: Optional[str] = None) -> "CheckpointRecord":
        data = dict(data or {})
        fmt = data.pop("fmt", data.pop("format", "pt"))
        return cls(
            env_id=str(data.get("env_id", "default")),
            kind=normalize_kind(data.get("kind", KIND_TARGET)),
            path=str(data.get("path", "")),
            seed=data.get("seed"),
            fmt=fmt,
            backend=data.get("backend"),
            timesteps=data.get("timesteps"),
            score=data.get("score"),
            created=float(data.get("created", time.time())),
            size_bytes=data.get("size_bytes"),
            tag=data.get("tag"),
            metadata=dict(data.get("metadata") or {}),
            root=root,
        )

    def format(self, decimals: int = 2) -> str:
        score = "n/a" if self.score is None else f"{float(self.score):.{decimals}f}"
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created))
        return (
            f"{self.env_id:<14} {self.kind:<11} seed={str(self.seed):<5} "
            f"score={score:<10} exists={str(self.exists):<5} {os.path.basename(self.path)} ({stamp})"
        )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class CheckpointManager:
    """Type-aware checkpoint store with a JSON registry.

    Parameters
    ----------
    root:
        Directory holding the checkpoints (created on demand).  Defaults to
        ``<project root>/policies``.
    registry:
        Registry filename (relative to ``root``) or absolute path.
    logger:
        Optional logger; a module logger is used otherwise.
    auto_register:
        Whether :meth:`save` records an entry in the registry (default ``True``).
    """

    def __init__(
        self,
        root: Optional[str] = None,
        registry: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        auto_register: bool = True,
    ) -> None:
        self.root = checkpoint_dir(root)
        if registry is None:
            self.registry_path = os.path.join(self.root, DEFAULT_REGISTRY_NAME)
        elif os.path.isabs(registry):
            self.registry_path = registry
        else:
            self.registry_path = os.path.join(self.root, registry)
        self.logger = logger or _LOGGER
        self.auto_register = bool(auto_register)

    # -------------------------------------------------------------- registry
    def load_registry(self) -> Dict[str, Any]:
        """Read the registry file (returns an empty skeleton when missing)."""
        raw = load_json(self.registry_path, default=None)
        if not isinstance(raw, dict):
            raw = {}
        records = raw.get("checkpoints")
        if not isinstance(records, list):
            records = []
        raw["checkpoints"] = records
        raw.setdefault("version", 1)
        raw["root"] = self.root
        return raw

    def save_registry(self, registry: Optional[Dict[str, Any]] = None) -> str:
        """Persist the registry file (best effort -- never raises)."""
        payload = registry if registry is not None else self._registry_cache()
        try:
            ensure_dir(os.path.dirname(self.registry_path))
            save_json(payload, self.registry_path)
        except Exception as exc:  # pragma: no cover - disk problems only
            self.logger.warning("Could not write checkpoint registry %s: %s", self.registry_path, exc)
        return self.registry_path

    def _registry_cache(self) -> Dict[str, Any]:
        return getattr(self, "_cache", {"version": 1, "checkpoints": [], "root": self.root})

    def records(self) -> List[CheckpointRecord]:
        """All registry entries as :class:`CheckpointRecord` objects."""
        raw = self.load_registry()
        out: List[CheckpointRecord] = []
        for entry in raw.get("checkpoints", []):
            try:
                out.append(CheckpointRecord.from_dict(entry, root=self.root))
            except Exception:
                continue
        return out

    def _write_all(self, records: Sequence[CheckpointRecord]) -> str:
        payload = {
            "version": 1,
            "root": self.root,
            "updated": time.time(),
            "checkpoints": [rec.to_dict() for rec in records],
        }
        self._cache = payload
        return self.save_registry(payload)

    def register(
        self,
        record: Any = None,
        env_id: Optional[Any] = None,
        kind: Optional[str] = None,
        path: Optional[str] = None,
        seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        replace: bool = True,
        **extra: Any,
    ) -> CheckpointRecord:
        """Insert/update a registry entry and flush the registry."""
        if isinstance(record, CheckpointRecord):
            rec = record
            rec.root = rec.root or self.root
        else:
            if env_id is None and isinstance(record, (str, os.PathLike)):
                path = path or str(record)
            rec = CheckpointRecord(
                env_id=normalize_env_key(env_id if env_id is not None else (record or "default")),
                kind=normalize_kind(kind),
                path=self._relativize(path) if path else "",
                seed=seed,
                fmt=os.path.splitext(str(path or ""))[1].lstrip(".") or "pt",
                metadata=dict(metadata or {}),
                root=self.root,
            )
        for key, value in (extra or {}).items():
            if hasattr(rec, key):
                setattr(rec, key, value)
            else:
                rec.metadata.setdefault(key, value)
        rec.refresh()

        existing = self.records()
        if replace:
            existing = [
                r
                for r in existing
                if not (
                    r.env_id == rec.env_id
                    and r.kind == rec.kind
                    and (r.seed or 0) == (rec.seed or 0)
                    and os.path.basename(r.path) == os.path.basename(rec.path)
                )
            ]
        existing.append(rec)
        self._write_all(existing)
        return rec

    def unregister(self, env_id: Any = None, kind: Optional[str] = None, seed: Optional[int] = None,
                   path: Optional[str] = None) -> int:
        """Drop matching entries; returns the number removed."""
        key = normalize_env_key(env_id) if env_id is not None else None
        kind_n = normalize_kind(kind) if kind is not None else None
        base = os.path.basename(path) if path else None
        kept: List[CheckpointRecord] = []
        removed = 0
        for rec in self.records():
            match = True
            if key is not None and rec.env_id != key:
                match = False
            if kind_n is not None and rec.kind != kind_n:
                match = False
            if seed is not None and (rec.seed or 0) != int(seed):
                match = False
            if base is not None and os.path.basename(rec.path) != base:
                match = False
            if match:
                removed += 1
            else:
                kept.append(rec)
        if removed:
            self._write_all(kept)
        return removed

    # ----------------------------------------------------------- paths/find
    def _relativize(self, path: Optional[str]) -> str:
        if not path:
            return ""
        try:
            abs_path = os.path.abspath(str(path))
            rel = os.path.relpath(abs_path, self.root)
            if not rel.startswith(".."):
                return rel
            return abs_path
        except Exception:
            return str(path)

    def path_for(self, env_id: Any, kind: str = KIND_TARGET, seed: Optional[int] = None,
                 tag: Optional[str] = None, ext: Optional[str] = None,
                 backend: Optional[str] = None) -> str:
        """Path inside this manager's root for ``(env, kind, seed)``."""
        return checkpoint_path(
            env_id, kind=kind, seed=seed, out_dir=self.root, tag=tag, ext=ext, backend=backend
        )

    def resolve(self, env_id: Any = None, kind: str = KIND_TARGET, seed: Optional[int] = None,
                path: Optional[str] = None, tag: Optional[str] = None) -> Optional[str]:
        """Resolve the file path of a checkpoint, registry first, then convention.

        Returns ``None`` when nothing exists on disk.
        """
        if path:
            candidate = path if os.path.isabs(path) else os.path.join(self.root, path)
            return candidate if os.path.isfile(candidate) else None

        key = normalize_env_key(env_id) if env_id is not None else None
        kind_n = normalize_kind(kind)

        candidates: List[CheckpointRecord] = []
        for rec in self.records():
            if key is not None and rec.env_id != key:
                continue
            if rec.kind != kind_n and normalize_kind(rec.kind) != kind_n:
                continue
            if seed is not None and (rec.seed or 0) != int(seed):
                continue
            if tag is not None and (rec.tag or "") != str(tag):
                continue
            if rec.exists:
                candidates.append(rec)
        if candidates:
            candidates.sort(key=lambda r: r.created, reverse=True)
            return candidates[0].abs_path

        if key is None:
            return None
        guessed = self.path_for(key, kind=kind_n, seed=seed, tag=tag)
        if os.path.isfile(guessed):
            return guessed
        # tolerate the alternative extension (.zip <-> .pt)
        alt = "".join(
            [os.path.splitext(guessed)[0], ".pt" if guessed.endswith(".zip") else ".zip"]
        )
        if os.path.isfile(alt):
            return alt
        return None

    def exists(self, env_id: Any, kind: str = KIND_TARGET, seed: Optional[int] = None) -> bool:
        return self.resolve(env_id=env_id, kind=kind, seed=seed) is not None

    def find(
        self,
        env_id: Any = None,
        kind: Optional[str] = None,
        seed: Optional[int] = None,
        pattern: Optional[str] = None,
    ) -> List[CheckpointRecord]:
        """Find registered (and unregistered-but-present) checkpoints."""
        key = normalize_env_key(env_id) if env_id is not None else None
        kind_n = normalize_kind(kind) if kind is not None else None
        found: List[CheckpointRecord] = []
        for rec in self.records():
            if key is not None and rec.env_id != key:
                continue
            if kind_n is not None and rec.kind != kind_n:
                continue
            if seed is not None and (rec.seed or 0) != int(seed):
                continue
            found.append(rec.refresh())

        if pattern:
            import glob

            for path in sorted(glob.glob(os.path.join(self.root, pattern))):
                if not os.path.isfile(path):
                    continue
                if any(r.abs_path == os.path.abspath(path) for r in found):
                    continue
                found.append(
                    CheckpointRecord(
                        env_id=key or normalize_env_key(os.path.basename(path)),
                        kind=kind_n or KIND_TARGET,
                        path=self._relativize(path),
                        fmt=os.path.splitext(path)[1].lstrip(".") or "pt",
                        root=self.root,
                    ).refresh()
                )
        found.sort(key=lambda r: (r.env_id, r.kind, r.seed if r.seed is not None else -1, r.created))
        return found

    def latest(self, env_id: Any, kind: str = KIND_TARGET) -> Optional[CheckpointRecord]:
        """Most recently created record for ``(env, kind)``."""
        records = [r for r in self.find(env_id=env_id, kind=kind) if r.exists]
        if not records:
            return None
        records.sort(key=lambda r: r.created, reverse=True)
        return records[0]

    # ---------------------------------------------------------------- save
    def save(
        self,
        obj: Any,
        env_id: Any,
        kind: str = KIND_TARGET,
        seed: Optional[int] = None,
        path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        backend: Optional[str] = None,
        tag: Optional[str] = None,
        timesteps: Optional[int] = None,
        score: Optional[float] = None,
        register: Optional[bool] = None,
        overwrite: bool = True,
        **extra: Any,
    ) -> str:
        """Persist ``obj`` using the right serializer for ``kind``.

        Returns the written path.  ``extra`` keywords are stored in the record
        metadata (e.g. ``alpha=1e-4``, ``p=0.5``, ``lam=0.01``, ``wall_time=...``).
        """
        kind_n = normalize_kind(kind)
        if path is None:
            path = self.path_for(env_id, kind=kind_n, seed=seed, tag=tag, backend=backend)
        path = path if os.path.isabs(path) else os.path.join(self.root, path)
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        if os.path.exists(path) and not overwrite:
            self.logger.info("Checkpoint already exists, keeping %s", path)
            return path

        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({k: v for k, v in (extra or {}).items() if v is not None})
        meta.setdefault("env_id", normalize_env_key(env_id))
        meta.setdefault("kind", kind_n)

        written_backend = self._save_any(obj, path, kind_n, backend=backend, meta=meta)

        rec = CheckpointRecord(
            env_id=normalize_env_key(env_id),
            kind=kind_n,
            path=self._relativize(path),
            seed=seed,
            fmt=os.path.splitext(path)[1].lstrip(".") or "pt",
            backend=written_backend,
            timesteps=timesteps if timesteps is not None else meta.get("timesteps"),
            score=score if score is not None else meta.get("score"),
            tag=tag,
            metadata=meta,
            root=self.root,
        ).refresh()

        resolved_register = self.auto_register if register is None else bool(register)
        if resolved_register:
            try:
                self.register(rec)
            except Exception as exc:  # pragma: no cover
                self.logger.warning("Could not register %s: %s", path, exc)
        self.logger.info("Saved %s checkpoint for %s at %s", kind_n, rec.env_id, path)
        return path

    def _save_any(
        self,
        obj: Any,
        path: str,
        kind: str,
        backend: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Type-aware save.  Returns a short backend tag."""
        meta = dict(meta or {})
        # 1) SB3 models expose .save() and .learn() / .policy
        if hasattr(obj, "save") and hasattr(obj, "learn") and not isinstance(obj, type):
            try:
                stem = path[: -len(os.path.splitext(path)[1])] if os.path.splitext(path)[1] else path
                obj.save(stem)
                written = stem + ".zip"
                if os.path.abspath(written) != os.path.abspath(path) and os.path.isfile(written):
                    pass
                return "sb3"
            except Exception as exc:
                self.logger.debug("SB3 save failed (%s); falling back to torch", exc)

        # 2) Stage-1 mask network / explanation artifacts
        if kind in (KIND_MASK, KIND_EXPLANATION) and save_mask_network is not None:
            try:
                save_mask_network(obj, path, env_id=meta.get("env_id"), **meta)
                return "torch"
            except Exception as exc:
                self.logger.debug("save_mask_network failed (%s); trying generic save", exc)

        # 3) Objects with their own .save(path, extra=...) -- RNDModule, PPORefiner,
        #    MaskTrainer, StateMaskTrainer, ExplanationHandle wrappers
        if hasattr(obj, "save") and not isinstance(obj, type):
            try:
                result = obj.save(path)
                if isinstance(result, str) and result:
                    return "native"
                return "native"
            except TypeError:
                try:
                    obj.save(path, extra=meta)  # type: ignore[call-arg]
                    return "native"
                except Exception as exc:
                    self.logger.debug("Object .save(path, extra=) failed (%s)", exc)
            except Exception as exc:
                self.logger.debug("Object .save(path) failed (%s)", exc)

        # 4) RICE policy helper
        if kind in (KIND_TARGET, KIND_REFINED) and save_policy is not None:
            try:
                save_policy(obj, path, env_id=meta.get("env_id"), extra=meta)
                return "rice-native"
            except Exception as exc:
                self.logger.debug("save_policy failed (%s); trying generic torch save", exc)

        # 5) Generic torch save (module, state_dict, dict payload)
        if _HAS_TORCH:  # pragma: no cover - depends on torch
            payload: Any = obj
            if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
                try:
                    payload = {"state_dict": obj.state_dict(), "metadata": meta}
                except Exception:
                    payload = obj
            torch.save(payload, path)
            return "torch"

        # 6) Last resort: JSON/pickle-free dump
        try:
            if isinstance(obj, (dict, list, tuple, str, int, float, bool)) or obj is None:
                save_json(obj, path)
                return "json"
        except Exception:
            pass
        raise RuntimeError(
            f"Cannot save object of type {type(obj)!r} for kind={kind!r} (no serializer available)"
        )

    # ---------------------------------------------------------------- load
    def load(
        self,
        env_id: Any = None,
        kind: str = KIND_TARGET,
        seed: Optional[int] = None,
        path: Optional[str] = None,
        build: bool = True,
        device: str = "cpu",
        observation_space: Any = None,
        action_space: Any = None,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        env: Any = None,
        policy: Any = None,
        map_location: Optional[str] = None,
        return_record: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Load a checkpoint (registry-aware).

        ``path`` short-circuits resolution; otherwise the newest matching entry
        wins and the conventional filename is tried last.  With
        ``return_record=True`` a ``(object, CheckpointRecord)`` tuple is
        returned in addition to the payload.
        """
        kind_n = normalize_kind(kind)
        resolved = self.resolve(env_id=env_id, kind=kind_n, seed=seed, path=path)
        if resolved is None:
            raise FileNotFoundError(
                f"No {kind_n} checkpoint found for env={env_id!r} seed={seed!r} under {self.root}"
            )

        obj = self._load_any(
            resolved,
            kind_n,
            env_id=env_id,
            build=build,
            device=device,
            observation_space=observation_space,
            action_space=action_space,
            obs_dim=obs_dim,
            action_dim=action_dim,
            env=env,
            policy=policy,
            map_location=map_location or device,
            **kwargs,
        )
        if not return_record:
            return obj
        record = self.latest(env_id or "", kind_n)
        return obj, record

    def _load_any(
        self,
        path: str,
        kind: str,
        env_id: Optional[Any] = None,
        build: bool = True,
        device: str = "cpu",
        observation_space: Any = None,
        action_space: Any = None,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        env: Any = None,
        policy: Any = None,
        map_location: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        ext = os.path.splitext(path)[1].lower()

        if kind in (KIND_MASK, KIND_EXPLANATION) and _HAS_MASK and build and load_mask_network is not None:
            try:
                return load_mask_network(
                    path,
                    env_id=env_id,
                    obs_dim=obs_dim,
                    observation_space=observation_space,
                    device=device,
                    **kwargs,
                )
            except Exception as exc:
                self.logger.debug("load_mask_network failed (%s); falling back", exc)

        if kind == KIND_RND and _HAS_RND and build and RNDModule is not None:
            try:
                return RNDModule.load(path, device=device, obs_dim=obs_dim, **kwargs)  # type: ignore[attr-defined]
            except Exception as exc:
                self.logger.debug("RNDModule.load failed (%s); falling back", exc)

        if kind == KIND_REFINER and _HAS_REFINER and build and PPORefiner is not None:
            try:
                return PPORefiner.load(path, env, **kwargs)  # type: ignore[attr-defined]
            except Exception as exc:
                self.logger.debug("PPORefiner.load failed (%s); falling back", exc)

        if kind in (KIND_TARGET, KIND_REFINED):
            if ext == ".zip":
                try:
                    from stable_baselines3 import PPO as _SB3PPO  # type: ignore

                    return _SB3PPO.load(path, device=device)
                except Exception as exc:
                    self.logger.debug("SB3 PPO.load failed for %s (%s)", path, exc)
            if _HAS_POLICIES and build and load_policy is not None:
                try:
                    return load_policy(
                        path,
                        env_id=env_id,
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        observation_space=observation_space,
                        action_space=action_space,
                        device=device,
                        **kwargs,
                    )
                except Exception as exc:
                    self.logger.debug("load_policy failed for %s (%s)", path, exc)

        if _HAS_TORCH:  # pragma: no cover - depends on torch
            return torch.load(path, map_location=map_location or "cpu")

        try:
            return load_json(path, default=None)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Cannot load checkpoint {path}: {exc}") from exc

    # Kind-specific conveniences -------------------------------------------
    def save_target(self, policy: Any, env_id: Any, **kwargs: Any) -> str:
        return self.save(policy, env_id, kind=KIND_TARGET, **kwargs)

    def load_target(self, env_id: Any, seed: Optional[int] = None, **kwargs: Any) -> Any:
        return self.load(env_id=env_id, kind=KIND_TARGET, seed=seed, **kwargs)

    def save_mask(self, mask_net: Any, env_id: Any, **kwargs: Any) -> str:
        return self.save(mask_net, env_id, kind=KIND_MASK, **kwargs)

    def load_mask(self, env_id: Any, seed: Optional[int] = None, **kwargs: Any) -> Any:
        return self.load(env_id=env_id, kind=KIND_MASK, seed=seed, **kwargs)

    def save_statemask(self, mask_net: Any, env_id: Any, **kwargs: Any) -> str:
        return self.save(mask_net, env_id, kind=KIND_STATEMASK, **kwargs)

    def load_statemask(self, env_id: Any, seed: Optional[int] = None, **kwargs: Any) -> Any:
        return self.load(env_id=env_id, kind=KIND_STATEMASK, seed=seed, **kwargs)

    def save_refined(self, policy: Any, env_id: Any, method: Optional[str] = None, **kwargs: Any) -> str:
        kwargs.setdefault("tag", method)
        return self.save(policy, env_id, kind=KIND_REFINED, **kwargs)

    def load_refined(self, env_id: Any, seed: Optional[int] = None, method: Optional[str] = None,
                     **kwargs: Any) -> Any:
        kwargs.setdefault("tag", method)
        return self.load(env_id=env_id, kind=KIND_REFINED, seed=seed, **kwargs)

    def save_rnd(self, rnd: Any, env_id: Any, **kwargs: Any) -> str:
        return self.save(rnd, env_id, kind=KIND_RND, **kwargs)

    def load_rnd(self, env_id: Any, seed: Optional[int] = None, **kwargs: Any) -> Any:
        return self.load(env_id=env_id, kind=KIND_RND, seed=seed, **kwargs)

    def save_refiner(self, refiner: Any, env_id: Any, **kwargs: Any) -> str:
        return self.save(refiner, env_id, kind=KIND_REFINER, **kwargs)

    def load_refiner(self, env_id: Any, seed: Optional[int] = None, **kwargs: Any) -> Any:
        return self.load(env_id=env_id, kind=KIND_REFINER, seed=seed, **kwargs)

    def save_explanation(self, handle: Any, env_id: Any, **kwargs: Any) -> Optional[str]:
        """Persist the mask network carried by an explanation handle (if any)."""
        mask_net = getattr(handle, "mask_net", None)
        if mask_net is None and hasattr(handle, "get_mask_net"):
            try:
                mask_net = handle.get_mask_net()
            except Exception:
                mask_net = None
        if mask_net is None:
            self.logger.debug("Explanation handle for %s has no mask net; nothing to save", env_id)
            return None
        metadata = kwargs.pop("metadata", {}) or {}
        method = getattr(handle, "method", None)
        if method:
            metadata.setdefault("explanation_method", method)
        train_time = getattr(handle, "train_time", None)
        if train_time is not None:
            metadata.setdefault("train_time", train_time)
        samples = getattr(handle, "samples", None)
        if samples is not None:
            metadata.setdefault("samples", samples)
        return self.save(mask_net, env_id, kind=KIND_MASK, metadata=metadata, **kwargs)

    # ------------------------------------------------------------- verify/prune
    def verify(self, env_id: Any = None, kind: Optional[str] = None,
               load: bool = False, **load_kwargs: Any) -> Dict[str, Any]:
        """Check registered checkpoints exist (optionally loadable)."""
        report: Dict[str, Any] = {
            "root": self.root,
            "registry": self.registry_path,
            "n_records": 0,
            "missing": [],
            "loaded": [],
            "failed": [],
            "ok": True,
        }
        records = self.find(env_id=env_id, kind=kind)
        report["n_records"] = len(records)
        for rec in records:
            if not rec.exists:
                report["missing"].append(rec.path)
                report["ok"] = False
                continue
            if not load:
                continue
            try:
                self._load_any(rec.abs_path, rec.kind, env_id=rec.env_id, build=False, **load_kwargs)
                report["loaded"].append(rec.path)
            except Exception as exc:
                report["failed"].append({"path": rec.path, "error": str(exc)})
        return report

    def prune(self, missing: bool = True, env_id: Any = None, kind: Optional[str] = None,
              delete: bool = False) -> Dict[str, Any]:
        """Drop registry entries whose files no longer exist.

        With ``delete=True`` the matching *files* are removed from disk instead
        (registry entries are dropped too).
        """
        removed: List[str] = []
        kept: List[CheckpointRecord] = []
        for rec in self.find(env_id=env_id, kind=kind):
            drop = False
            if delete and rec.exists:
                try:
                    os.remove(rec.abs_path)
                    drop = True
                except OSError as exc:
                    self.logger.warning("Could not delete %s: %s", rec.abs_path, exc)
            elif missing and not rec.exists:
                drop = True
            if drop:
                removed.append(rec.path)
            else:
                kept.append(rec)
        self._write_all(kept)
        return {"removed": removed, "remaining": len(kept), "root": self.root}

    # ------------------------------------------------------------- reporting
    def list(self, kind: Optional[str] = None, env: Any = None) -> List[CheckpointRecord]:
        """Alias of :meth:`find` (name parity with the CLI)."""
        return self.find(env_id=env, kind=kind)

    def index(self) -> Dict[str, Any]:
        """Rebuild the registry from the files present in ``root``."""
        import glob

        found: List[CheckpointRecord] = []
        for path in sorted(glob.glob(os.path.join(self.root, "**", "*"), recursive=True)):
            if not os.path.isfile(path):
                continue
            if os.path.basename(path) == DEFAULT_REGISTRY_NAME:
                continue
            stem = os.path.splitext(os.path.basename(path))[0]
            kind = KIND_TARGET
            for candidate, stem_name in STEMS.items():
                if stem.endswith(stem_name):
                    kind = candidate
                    if stem_name in ("ppo", "mask", "statemask", "refined", "rnd", "refiner"):
                        break
            key = normalize_env_key(stem)
            for candidate in sorted(STEMS.values(), key=len, reverse=True):
                if key.endswith(candidate):
                    key = key[: -len(candidate)].rstrip("_") or key
            seed = None
            if "_seed" in stem:
                tail = stem.rsplit("_seed", 1)[-1]
                if tail.isdigit():
                    seed = int(tail)
            found.append(
                CheckpointRecord(
                    env_id=key,
                    kind=normalize_kind(kind),
                    path=self._relativize(path),
                    seed=seed,
                    fmt=os.path.splitext(path)[1].lstrip(".") or "pt",
                    root=self.root,
                ).refresh()
            )
        self._write_all(found)
        return {"root": self.root, "indexed": len(found), "registry": self.registry_path}

    def summary(self) -> Dict[str, Any]:
        """Aggregate view of the checkpoint store."""
        records = self.find()
        by_kind: Dict[str, int] = {}
        by_env: Dict[str, int] = {}
        present = 0
        for rec in records:
            by_kind[rec.kind] = by_kind.get(rec.kind, 0) + 1
            by_env[rec.env_id] = by_env.get(rec.env_id, 0) + 1
            present += int(rec.exists)
        return {
            "root": self.root,
            "registry": self.registry_path,
            "n_checkpoints": len(records),
            "n_present": present,
            "by_kind": by_kind,
            "by_env": by_env,
            "records": [rec.to_dict() for rec in records],
        }

    def describe(self) -> str:
        info = self.summary()
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(info["by_kind"].items())) or "none"
        return (
            f"CheckpointManager(root={info['root']}, n={info['n_checkpoints']} "
            f"({info['n_present']} on disk), kinds: {kinds})"
        )

    def format(self, kind: Optional[str] = None, env: Any = None, decimals: int = 2) -> str:
        records = self.find(env_id=env, kind=kind)
        if not records:
            return f"No checkpoints under {self.root}"
        lines = [self.describe(), ""]
        lines.extend(rec.format(decimals=decimals) for rec in records)
        return "\n".join(lines)

    # --------------------------------------------------- object introspection
    @staticmethod
    def detect_kind(obj: Any) -> str:
        return detect_kind(obj)


# ---------------------------------------------------------------------------
# Module-level helpers (thin wrappers for script / driver use)
# ---------------------------------------------------------------------------
def build_manager(root: Optional[str] = None, registry: Optional[str] = None,
                  logger: Optional[logging.Logger] = None) -> CheckpointManager:
    """Factory mirroring the rest of the RICE module surface."""
    return CheckpointManager(root=root, registry=registry, logger=logger)


def detect_kind(obj: Any) -> str:
    """Best-effort classification of a RICE artifact object."""
    if obj is None:
        return KIND_TARGET
    name = type(obj).__name__.lower()
    if _HAS_RND and RNDModule is not None and isinstance(obj, RNDModule):  # type: ignore[arg-type]
        return KIND_RND
    if _HAS_REFINER and PPORefiner is not None and isinstance(obj, PPORefiner):  # type: ignore[arg-type]
        return KIND_REFINER
    if _HAS_MASK_TRAINER and MaskTrainer is not None and isinstance(obj, MaskTrainer):  # type: ignore[arg-type]
        return KIND_REFINER
    if "mask" in name and "network" in name:
        return KIND_MASK
    if "rnd" in name:
        return KIND_RND
    if "refiner" in name or "trainer" in name:
        return KIND_REFINER
    if "statemask" in name:
        return KIND_STATEMASK
    if hasattr(obj, "learn") and hasattr(obj, "save"):  # SB3 model
        return KIND_TARGET
    if hasattr(obj, "target") and hasattr(obj, "predictor"):
        return KIND_RND
    if hasattr(obj, "policy") and hasattr(obj, "pretrained_policy"):
        return KIND_REFINER
    if "policy" in name or "actorcritic" in name:
        return KIND_TARGET
    return KIND_TARGET


def _manager_for(root: Optional[str] = None, manager: Optional[CheckpointManager] = None) -> CheckpointManager:
    return manager if manager is not None else CheckpointManager(root=root)


def save_checkpoint(obj: Any, env_id: Any, kind: Optional[str] = None, seed: Optional[int] = None,
                    path: Optional[str] = None, out_dir: Optional[str] = None,
                    manager: Optional[CheckpointManager] = None,
                    metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:
    """Save ``obj`` for ``env_id``; ``kind`` is auto-detected when omitted."""
    mgr = _manager_for(out_dir, manager)
    return mgr.save(
        obj,
        env_id,
        kind=kind or detect_kind(obj),
        seed=seed,
        path=path,
        metadata=metadata,
        **kwargs,
    )


def load_checkpoint(env_id: Any = None, kind: str = KIND_TARGET, seed: Optional[int] = None,
                    path: Optional[str] = None, out_dir: Optional[str] = None,
                    manager: Optional[CheckpointManager] = None, **kwargs: Any) -> Any:
    """Load a checkpoint (see :meth:`CheckpointManager.load`)."""
    mgr = _manager_for(out_dir, manager)
    return mgr.load(env_id=env_id, kind=kind, seed=seed, path=path, **kwargs)


def resolve_checkpoint(env_id: Any = None, kind: str = KIND_TARGET, seed: Optional[int] = None,
                       path: Optional[str] = None, out_dir: Optional[str] = None,
                       manager: Optional[CheckpointManager] = None) -> Optional[str]:
    """Path of a checkpoint, or ``None`` when it has not been produced yet."""
    mgr = _manager_for(out_dir, manager)
    return mgr.resolve(env_id=env_id, kind=kind, seed=seed, path=path)


def list_checkpoints(env_id: Any = None, kind: Optional[str] = None, out_dir: Optional[str] = None,
                     manager: Optional[CheckpointManager] = None) -> List[CheckpointRecord]:
    """All checkpoints, optionally filtered by env/kind."""
    mgr = _manager_for(out_dir, manager)
    return mgr.find(env_id=env_id, kind=kind)


def index_checkpoints(out_dir: Optional[str] = None,
                      manager: Optional[CheckpointManager] = None) -> Dict[str, Any]:
    """Rebuild the registry from files on disk."""
    return _manager_for(out_dir, manager).index()


def prune_checkpoints(missing: bool = True, delete: bool = False, out_dir: Optional[str] = None,
                      manager: Optional[CheckpointManager] = None) -> Dict[str, Any]:
    """Remove stale (or, with ``delete=True``, existing) checkpoint records."""
    return _manager_for(out_dir, manager).prune(missing=missing, delete=delete)


def read_registry(out_dir: Optional[str] = None) -> Dict[str, Any]:
    """Read the JSON registry without instantiating a full manager."""
    root = checkpoint_dir(out_dir)
    return load_json(os.path.join(root, DEFAULT_REGISTRY_NAME), default={"version": 1, "checkpoints": []})


def write_registry(registry: Dict[str, Any], out_dir: Optional[str] = None) -> str:
    """Write the JSON registry."""
    root = checkpoint_dir(out_dir)
    path = os.path.join(root, DEFAULT_REGISTRY_NAME)
    save_json(registry, path)
    return path


def describe_manager(manager: Optional[CheckpointManager] = None) -> str:
    """One-line availability description for logging."""
    mgr = manager or CheckpointManager()
    return mgr.describe()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser():  # pragma: no cover - CLI surface
    import argparse

    parser = argparse.ArgumentParser(
        prog="checkpoint_manager",
        description=(
            "Inspect / index the RICE checkpoint store. Canonical layout: "
            "policies/<env>_ppo[_seedN].zip (frozen pi), "
            "policies/<env>_mask[_seedN].pt (Stage-1 mask net), "
            "policies/<env>_refined[_seedN].pt (Stage-2 pi')."
        ),
    )
    parser.add_argument("command", nargs="?", default="list",
                        choices=["list", "index", "summary", "resolve", "verify", "prune", "registry"])
    parser.add_argument("--env", "--env-id", dest="env", default=None, type=str)
    parser.add_argument("--kind", default=None, type=str,
                        choices=list(KINDS) + [None])
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--path", default=None, type=str)
    parser.add_argument("--root", "--out-dir", dest="root", default=None, type=str)
    parser.add_argument("--delete", action="store_true",
                        help="for 'prune': delete the files themselves")
    parser.add_argument("--all", action="store_true",
                        help="for 'prune': purge entries even when the file exists")
    parser.add_argument("--json", action="store_true", help="emit JSON where applicable")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI surface
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    )
    manager = CheckpointManager(root=args.root)
    command = args.command

    if command == "list":
        if args.json:
            print(json.dumps([r.to_dict() for r in manager.find(env_id=args.env, kind=args.kind)],
                             indent=2, default=str))
        else:
            print(manager.format(kind=args.kind, env=args.env))
        return 0

    if command == "summary":
        info = manager.summary()
        if args.json:
            print(json.dumps(info, indent=2, default=str))
        else:
            print(manager.describe())
            for rec in manager.find(env_id=args.env, kind=args.kind):
                print("  " + rec.format())
        return 0

    if command == "index":
        info = manager.index()
        print(json.dumps(info, indent=2, default=str) if args.json else
              f"Indexed {info['indexed']} checkpoints into {info['registry']}")
        return 0

    if command == "resolve":
        path = manager.resolve(env_id=args.env, kind=args.kind or KIND_TARGET, seed=args.seed, path=args.path)
        if path is None:
            print("not found")
            return 1
        print(path)
        return 0

    if command == "verify":
        report = manager.verify(env_id=args.env, kind=args.kind, load=True)
        print(json.dumps(report, indent=2, default=str) if args.json else
              f"records={report['n_records']} missing={len(report['missing'])} "
              f"failed_to_load={len(report['failed'])}")
        return 0 if report.get("ok", True) else 1

    if command == "prune":
        report = manager.prune(missing=not args.all, delete=args.delete, env_id=args.env, kind=args.kind)
        print(json.dumps(report, indent=2, default=str) if args.json else
              f"removed={len(report['removed'])} remaining={report['remaining']}")
        return 0

    if command == "registry":
        registry = manager.load_registry()
        print(json.dumps(registry, indent=2, default=str) if args.json else
              f"{len(registry.get('checkpoints', []))} entries in {manager.registry_path}")
        return 0

    parser.error(f"unknown command {command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
