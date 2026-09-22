"""NLD-AA dataset pipeline for the NetHack experiments (Appendix B.1).

This module reproduces the data column of the NetHack track of
*"Fine-tuning Reinforcement Learning Models is Secretly a Forgetting
Mitigation Problem"* (Wolczyk et al., 2024):

* **Download** the 16 shards of the *NetHack Learning Dataset* subset
  ``NLD-AA`` (~8 000 Human Monk games) from
  ``https://dl.fbaipublicfiles.com/nld/nld-aa/``.
* **Build/register the local sqlite database** named ``nld-aa-v0`` using
  ``nle.dataset`` (the official NetHack Learning Environment loader).
* Expose a :class:`TtyrecDataset` iterator with the paper's mini-batch size
  of ``128`` (Appendix B.1), used for three purposes:
  1. building the behavioral-cloning state buffer ``B_BC`` (10 000 states,
     Appendix C.2 / Table 3),
  2. producing the batches from which the diagonal Fisher information matrix
     of the actor is accumulated (10 000 batches, Appendix B.1/C.1),
  3. computing expert-action log-likelihoods.

Everything in this file degrades gracefully: the module imports fine without
``nle`` (or without the downloaded shards), and the heavy operations raise a
clear, actionable :class:`DatasetUnavailableError`.

Shard naming
------------
The public NLD-AA distribution has been published under slightly different
file-name conventions over time (``nld-aa-v0-00000.tar``,
``nld-aa-v0.tar.gz.aa``, ...).  :data:`SHARD_PATTERNS` therefore lists the
candidate patterns that are probed (in order) against the server and only the
first successful one is used; ``--pattern`` lets the user force a specific
convention.  This keeps the downloader working regardless of the exact naming
of the mirror.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Optional heavy dependencies (numpy / torch / nle)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - numpy is a documented requirement
    import numpy as np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

try:  # pragma: no cover - torch only needed for tensors
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Constants (Appendix B.1 - Dataset)
# ---------------------------------------------------------------------------
NLD_AA_URL_BASE = "https://dl.fbaipublicfiles.com/nld/nld-aa/"
DATASET_NAME = "nld-aa-v0"
NUM_SHARDS = 16                       # 16 NLD-AA shards (Appendix B.1)
NUM_EPISODES = 8_000                  # ~8 000 Human Monk games
DEFAULT_BATCH_SIZE = 128              # NLD-AA mini-batch size (Table 1/B.1)
DEFAULT_SEQ_LENGTH = 1                # per-timestep batches for BC / Fisher
NUM_FISHER_BATCHES = 10_000           # Fisher accumulation (Appendix C.1)
EXPERIMENT_NAME = "nld-aa"
LMDB_EXT = ".db"

#: Candidate shard file-name patterns, ``{i}`` is the 0-based shard index.
#: The downloader probes them in order and keeps the first that responds.
SHARD_PATTERNS: Tuple[str, ...] = (
    "nld-aa-v0-{i}.tar",
    "nld-aa-v0-{i:02d}.tar",
    "nld-aa-v0-{i:05d}.tar",
    "nld-aa-v0.tar.{aa}",
    "nld-aa-v0.tar.gz.{aa}",
    "nld-aa-v0-{i}.tar.gz",
    "nld-aa-v0-{i:02d}.tar.gz",
)

#: Observation keys consumed by the NetHack model (Appendix B.1).
OBSERVATION_KEYS: Tuple[str, ...] = ("tty_chars", "tty_colors", "blstats", "message")

#: Aliases accepted when reading batches returned by ``nle.dataset``.
_CHAR_KEYS = ("tty_chars", "chars", "tty_records")
_COLOR_KEYS = ("tty_colors", "colors")
_BLSTATS_KEYS = ("blstats", "bl_stats", "tty_blstats")
_MESSAGE_KEYS = ("message", "messages", "tty_message")
_ACTION_KEYS = ("actions", "action", "tty_actions")
_GAMEID_KEYS = ("gameids", "game_id", "gameid", "episode_ids")

#: Small helper: the ASCII-suffix naming (``.aa``, ``.ab``, ...) needs letters.
_AA_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


class DatasetUnavailableError(RuntimeError):
    """Raised when the NLD-AA data (or ``nle``) is not available locally."""


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def default_data_dir() -> str:
    """Return the default directory holding downloaded shards / sqlite db."""
    env = os.environ.get("NLD_AA_DATA", os.environ.get("NETHACK_DATA"))
    if env:
        return env
    local = "data/nld_aa"
    if os.path.isdir(local):
        return local
    return os.path.join(os.path.expanduser("~"), ".nld_aa")


def aa_suffix(index: int) -> str:
    """``0 -> 'aa'``, ``1 -> 'ab'`` ... mirroring ``split -b`` output names."""
    first = _AA_ALPHABET[(index // len(_AA_ALPHABET)) % len(_AA_ALPHABET)]
    second = _AA_ALPHABET[index % len(_AA_ALPHABET)]
    return first + second


def shard_names(
    num_shards: int = NUM_SHARDS,
    pattern: Optional[str] = None,
    patterns: Sequence[str] = SHARD_PATTERNS,
) -> List[str]:
    """Return the candidate file names for ``num_shards`` NLD-AA shards."""
    pats = (pattern,) if pattern else tuple(patterns)
    names: List[str] = []
    for i in range(num_shards):
        formatted = []
        for pat in pats:
            formatted.append(pat.format(i=i, aa=aa_suffix(i)))
        names.append(os.path.join(*formatted) if False else formatted[0])
    return names


def shard_candidates(
    num_shards: int = NUM_SHARDS, patterns: Sequence[str] = SHARD_PATTERNS
) -> List[List[str]]:
    """Return, per shard, the full list of candidate names to probe."""
    out: List[List[str]] = []
    for i in range(num_shards):
        out.append([pat.format(i=i, aa=aa_suffix(i)) for pat in patterns])
    return out


def shard_urls(
    url_base: str = NLD_AA_URL_BASE, num_shards: int = NUM_SHARDS, pattern: Optional[str] = None
) -> List[str]:
    """Full URLs of the NLD-AA shards."""
    if pattern:
        return [url_base + pattern.format(i=i, aa=aa_suffix(i)) for i in range(num_shards)]
    return [url_base + n for n in shard_names(num_shards=num_shards)]


def human_bytes(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PiB"


def _log(message: str) -> None:
    print(f"[nld-aa] {message}", flush=True)


# ---------------------------------------------------------------------------
# nle.dataset bridge
# ---------------------------------------------------------------------------
def nle_dataset_module() -> Optional[Any]:
    """Import ``nle.dataset`` lazily; return ``None`` when unavailable."""
    try:  # pragma: no cover - depends on the installation
        import nle.dataset as nle_dataset  # type: ignore

        return nle_dataset
    except Exception:
        pass
    try:  # pragma: no cover
        from nle import dataset as nle_dataset  # type: ignore

        return nle_dataset
    except Exception:
        return None


def nle_available() -> bool:
    """``True`` when the NetHack Learning Environment dataset tools exist."""
    return nle_dataset_module() is not None


def nle_db_dir() -> str:
    """Directory where ``nle.dataset`` stores built sqlite databases."""
    mod = nle_dataset_module()
    if mod is not None:
        for attr in ("db_path", "DB_PATH", "db_dir"):
            value = getattr(mod, attr, None)
            if isinstance(value, str) and value:
                return value
    return os.path.join(os.path.expanduser("~"), ".nle", "db")


def database_path(name: str = DATASET_NAME, data_dir: Optional[str] = None) -> str:
    """Path of the local sqlite database for ``name``."""
    if data_dir is not None:
        return os.path.join(data_dir, f"{name}{LMDB_EXT}")
    local = default_data_dir()
    candidate = os.path.join(local, f"{name}{LMDB_EXT}")
    if os.path.exists(candidate):
        return candidate
    nle_path = os.path.join(nle_db_dir(), f"{name}{LMDB_EXT}")
    if os.path.exists(nle_path) or not os.path.exists(candidate):
        return nle_path
    return candidate


# ---------------------------------------------------------------------------
# Downloading / unpacking shards
# ---------------------------------------------------------------------------
def download_shards(
    dest: Optional[str] = None,
    url_base: str = NLD_AA_URL_BASE,
    num_shards: int = NUM_SHARDS,
    pattern: Optional[str] = None,
    force: bool = False,
    probe: bool = True,
    timeout: float = 30.0,
    log: bool = True,
) -> List[str]:
    """Download the ``num_shards`` NLD-AA shards into ``dest``.

    Parameters
    ----------
    dest:
        Target directory (defaults to :func:`default_data_dir`).
    probe:
        When ``True`` (default) every candidate name from
        :data:`SHARD_PATTERNS` is HEAD-probed and the first that exists is
        used.  This makes the downloader robust to the exact mirror naming.

    Returns
    -------
    list of str
        Paths of the successfully downloaded files (already-existing files
        are returned without re-downloading unless ``force=True``).
    """
    import urllib.error
    import urllib.request

    dest = dest or default_data_dir()
    os.makedirs(dest, exist_ok=True)

    resolved: List[str] = []
    for shard, candidates in enumerate(shard_candidates(num_shards=num_shards)):
        if pattern:
            candidates = [pattern.format(i=shard, aa=aa_suffix(shard))]
        chosen = None
        if not probe:
            chosen = candidates[0]
        else:
            for name in candidates:
                url = url_base + name
                try:
                    request = urllib.request.Request(url, method="HEAD")
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        if 200 <= getattr(response, "status", 200) < 300:
                            chosen = name
                            break
                except Exception:
                    continue
        if chosen is None:
            chosen = candidates[0]
        target = os.path.join(dest, os.path.basename(chosen))
        resolved.append(target)
        if os.path.exists(target) and not force and os.path.getsize(target) > 0:
            if log:
                _log(f"shard {shard:02d} already present: {os.path.basename(target)}")
            continue
        url = url_base + chosen
        if log:
            _log(f"downloading shard {shard:02d}/{num_shards}: {url}")
        _download(url, target, log=log)
    return resolved


def _download(url: str, target: str, log: bool = True, chunk_size: int = 1 << 20) -> str:
    """Stream ``url`` to ``target`` with a tiny progress meter."""
    import urllib.request

    tmp = target + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "nld-aa-downloader/1.0"})
    with urllib.request.urlopen(request) as response, open(tmp, "wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        last_report = 0.0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if log and now - last_report > 2.0:
                last_report = now
                if total:
                    _log(f"  {human_bytes(downloaded)} / {human_bytes(total)}")
                else:
                    _log(f"  {human_bytes(downloaded)}")
    os.replace(tmp, target)
    return target


def unpack_shards(
    shard_paths: Sequence[str],
    dest: Optional[str] = None,
    delete: bool = False,
    log: bool = True,
) -> List[str]:
    """Extract ``.tar``/``.tar.gz``/``.zip`` shards (``.aa``/``.tar.aa`` etc.).

    Files that are already plain ttyrec or sqlite files are returned as-is.
    """
    import tarfile
    import zipfile

    dest = dest or os.path.dirname(os.path.abspath(shard_paths[0])) if shard_paths else default_data_dir()
    os.makedirs(dest, exist_ok=True)
    extracted: List[str] = []

    for path in shard_paths:
        if not path:
            continue
        base = os.path.basename(path)
        try:
            if base.endswith((".tar", ".tar.gz", ".tgz")) or ".tar.aa" in base or ".tar.gz." in base:
                with tarfile.open(path, "r:*") as tar:
                    if log:
                        _log(f"extracting {base} -> {dest}")
                    tar.extractall(dest)
                    extracted.append(dest)
            elif base.endswith(".zip"):
                with zipfile.ZipFile(path) as zf:
                    if log:
                        _log(f"extracting {base} -> {dest}")
                    zf.extractall(dest)
                    extracted.append(dest)
            else:
                extracted.append(path)
        except Exception as exc:  # pragma: no cover - depends on data
            if log:
                _log(f"could not extract {base}: {exc}")
            extracted.append(path)
        if delete and os.path.exists(path):
            os.remove(path)
    return extracted


def find_ttyrec_files(directory: Optional[str] = None, recursive: bool = True) -> List[str]:
    """Locate local ``.ttyrec`` / ``.ttyrec.bz2`` / ``.ttyrec.gz`` files."""
    directory = directory or default_data_dir()
    files: List[str] = []
    for suffix in ("*.ttyrec", "*.ttyrec.bz2", "*.ttyrec.gz", "*.ttyrec.xz"):
        files.extend(glob.glob(os.path.join(directory, "**", suffix), recursive=recursive))
    return sorted(files)


# ---------------------------------------------------------------------------
# Building / registering the sqlite database
# ---------------------------------------------------------------------------
def add_nld_aa_datafiles(files: Sequence[str], log: bool = True) -> Dict[str, Any]:
    """Register local ttyrec files with ``nle.dataset`` (official helper)."""
    mod = nle_dataset_module()
    if mod is None:
        raise DatasetUnavailableError(
            "nle.dataset is unavailable; install the NetHack Learning Environment (pip install nle)."
        )
    files = [f for f in files if f]
    if hasattr(mod, "add_nld_aa_datafiles"):
        if log:
            _log(f"registering {len(files)} ttyrec files via nle.dataset.add_nld_aa_datafiles")
        try:
            return {"result": mod.add_nld_aa_datafiles(files)}
        except TypeError:
            # Some NLE versions expect a single list argument of directories.
            directories = sorted({os.path.dirname(f) or "." for f in files})
            return {"result": mod.add_nld_aa_datafiles(directories)}
    if hasattr(mod, "add_ttyrec_datafiles"):
        return {"result": mod.add_ttyrec_datafiles(files)}
    return {}


def add_dataset(path: str, name: str = DATASET_NAME) -> Any:
    """Add a built sqlite database to ``nle.dataset`` under ``name``."""
    mod = nle_dataset_module()
    if mod is None:
        raise DatasetUnavailableError("nle.dataset is unavailable; install nle to register datasets.")
    if hasattr(mod, "add_dataset"):
        return mod.add_dataset(path, name)
    raise DatasetUnavailableError("nle.dataset has no 'add_dataset' entry point in this version.")


def build_dataset(
    data_dir: Optional[str] = None,
    name: str = DATASET_NAME,
    files: Optional[Sequence[str]] = None,
    num_shards: int = NUM_SHARDS,
    download: bool = False,
    force: bool = False,
    register: bool = True,
    log: bool = True,
) -> str:
    """Build (and optionally download/register) the local NLD-AA database.

    Returns the path of the sqlite database.
    """
    data_dir = data_dir or default_data_dir()
    os.makedirs(data_dir, exist_ok=True)

    if files is None:
        files = find_ttyrec_files(data_dir)
    if not files and download:
        shards = download_shards(dest=data_dir, num_shards=num_shards, force=force, log=log)
        unpack_shards(shards, dest=data_dir, log=log)
        files = find_ttyrec_files(data_dir)
    if not files:
        raise DatasetUnavailableError(
            f"no ttyrec files found in {data_dir!r}; run download_shards(...)/unpack_shards(...) "
            "or pass files=..."
        )

    add_nld_aa_datafiles(files, log=log)
    db = database_path(name, data_dir=data_dir)
    os.makedirs(os.path.dirname(db) or ".", exist_ok=True)
    mod = nle_dataset_module()
    if mod is not None and hasattr(mod, "build_dataset"):
        if log:
            _log(f"building sqlite database {db}")
        # Different NLE versions expose different signatures; try the common ones.
        for kwargs in (
            {"name": name, "files": files, "path": db},
            {"name": name, "path": db},
            {"dataset_name": name, "path": db},
        ):
            try:
                built = mod.build_dataset(**kwargs)  # type: ignore[arg-type]
                db = built or db
                break
            except TypeError:
                continue
            except Exception as exc:  # pragma: no cover
                if log:
                    _log(f"build_dataset failed ({exc}); assuming nle keeps its own database")
                break
    if register:
        try:
            add_dataset(db, name)
        except DatasetUnavailableError:
            pass
    return db


def dataset_exists(name: str = DATASET_NAME, data_dir: Optional[str] = None) -> bool:
    """``True`` when a local database or ``nle`` registration is available."""
    path = database_path(name, data_dir=data_dir)
    if os.path.exists(path):
        return True
    mod = nle_dataset_module()
    if mod is None:
        return False
    try:
        datasets = mod.datasets  # type: ignore[attr-defined]
        if isinstance(datasets, Mapping):
            return name in datasets
        return name in set(datasets)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# TtyrecDataset iterator
# ---------------------------------------------------------------------------
def _resolve_attr(value: Any, keys: Sequence[str]) -> Any:
    for key in keys:
        if isinstance(value, Mapping) and key in value:
            return value[key]
        if hasattr(value, key):
            return getattr(value, key)
    return None


@dataclass
class BatchSpec:
    """Description of an iteration over the NLD-AA dataset."""

    name: str = DATASET_NAME
    batch_size: int = DEFAULT_BATCH_SIZE
    seq_length: int = DEFAULT_SEQ_LENGTH
    num_batches: Optional[int] = None
    shuffle: bool = True
    loop_forever: bool = True
    max_frames_per_episode: Optional[int] = None
    seed: Optional[int] = None
    observation_keys: Tuple[str, ...] = OBSERVATION_KEYS

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TtyrecDataset:
    """Iterator over mini-batches of the registered ``nld-aa-v0`` dataset.

    Thin, dependency-light wrapper over ``nle.dataset.TtyrecDataset`` that

    * normalises the returned batch into a plain ``dict`` of arrays/tensors,
    * exposes the observation keys used by the NetHack model,
    * supports a bounded number of batches (``num_batches``) as required by
      the Fisher accumulation (10 000 batches) and the BC buffer (10 000
      states).

    When ``nle`` (or the database) is missing the iterator raises
    :class:`DatasetUnavailableError` on first use, so callers may still build
    configs/plans offline.
    """

    def __init__(
        self,
        name: str = DATASET_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seq_length: int = DEFAULT_SEQ_LENGTH,
        num_batches: Optional[int] = None,
        shuffle: bool = True,
        loop_forever: bool = True,
        max_frames_per_episode: Optional[int] = None,
        seed: Optional[int] = None,
        data_dir: Optional[str] = None,
        observation_keys: Sequence[str] = OBSERVATION_KEYS,
        device: Optional[str] = None,
        ttyrec_dataset: Any = None,
        dataset_module: Any = None,
    ) -> None:
        self.name = name
        self.batch_size = int(batch_size)
        self.seq_length = int(seq_length)
        self.num_batches = None if num_batches is None else int(num_batches)
        self.shuffle = bool(shuffle)
        self.loop_forever = bool(loop_forever)
        self.max_frames_per_episode = max_frames_per_episode
        self.seed = seed
        self.data_dir = data_dir
        self.observation_keys = tuple(observation_keys)
        self.device = device
        self._iterator = ttyrec_dataset
        self._module = dataset_module
        self._produced = 0
        self._generator = None
        if seed is not None and _HAS_TORCH and self.device and str(self.device).startswith("cuda"):
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(int(seed))

    # -- construction helpers -------------------------------------------------
    @property
    def module(self) -> Any:
        if self._module is None:
            self._module = nle_dataset_module()
        return self._module

    def _ensure_iterator(self) -> Any:
        if self._iterator is not None:
            return self._iterator
        mod = self.module
        if mod is None:
            raise DatasetUnavailableError(
                "nle.dataset is unavailable; install the NetHack Learning Environment to iterate "
                "over NLD-AA.  (pip install nle, then scripts/download_nld_aa.sh)"
            )
        if not dataset_exists(self.name, data_dir=self.data_dir):
            db = database_path(self.name, data_dir=self.data_dir)
            if os.path.exists(db):
                try:
                    add_dataset(db, self.name)
                except DatasetUnavailableError:
                    pass
            else:
                raise DatasetUnavailableError(
                    f"dataset {self.name!r} is not registered and no database found at {db!r}; "
                    "run scripts/download_nld_aa.sh first."
                )
        kwargs: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "shuffle": self.shuffle,
            "loop_forever": self.loop_forever,
        }
        if self.max_frames_per_episode is not None:
            kwargs["max_frames_per_episode"] = self.max_frames_per_episode
        if self.device is not None:
            kwargs["device"] = self.device
        for extra in ("seed", "generator"):
            if self.seed is not None and extra == "seed":
                kwargs["seed"] = int(self.seed)
            if self._generator is not None and extra == "generator":
                kwargs["generator"] = self._generator
        cls = getattr(mod, "TtyrecDataset")
        for attempt in (kwargs, {k: v for k, v in kwargs.items() if k != "device"}):
            try:
                self._iterator = cls(self.name, **attempt)  # type: ignore[call-arg]
                break
            except TypeError:
                continue
        if self._iterator is None:
            self._iterator = cls(self.name, self.batch_size, self.seq_length, self.shuffle)
        return self._iterator

    # -- iteration ------------------------------------------------------------
    def __iter__(self) -> "TtyrecDataset":
        self._produced = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        if self.num_batches is not None and self._produced >= self.num_batches:
            raise StopIteration
        iterator = self._ensure_iterator()
        batch = next(iterator)
        self._produced += 1
        return normalise_batch(batch)

    def next_batch(self) -> Dict[str, Any]:
        """Explicit ``__next__`` (handy when ``next()`` on a raw iterator)."""
        return self.__next__()

    def __len__(self) -> int:
        if self.num_batches is not None:
            return self.num_batches
        return NUM_FISHER_BATCHES if self.seq_length <= 1 else math.inf  # type: ignore[return-value]

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"TtyrecDataset(name={self.name!r}, batch_size={self.batch_size}, "
            f"seq_length={self.seq_length}, num_batches={self.num_batches}, shuffle={self.shuffle})"
        )


# ---------------------------------------------------------------------------
# Batch normalisation / extraction
# ---------------------------------------------------------------------------
def normalise_batch(batch: Any) -> Dict[str, Any]:
    """Convert an ``nle.dataset`` batch into a plain dict of numpy arrays.

    Handles mapping-style batches (usual case), attribute-style batches and
    ``(observations, actions)`` tuples.  ``numpy`` conversion is best-effort so
    torch tensors simply pass through.
    """
    out: Dict[str, Any] = {}
    if batch is None:
        return out
    if isinstance(batch, Mapping):
        for key, value in batch.items():
            out[str(key)] = _as_array(value)
        return out
    if _HAS_NUMPY and isinstance(batch, np.ndarray):
        return {"observations": batch}
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        obs, actions = batch
        if isinstance(obs, Mapping):
            out.update({str(k): _as_array(v) for k, v in obs.items()})
        else:
            out["observations"] = _as_array(obs)
        out["actions"] = _as_array(actions)
        return out
    # attribute-style (e.g. a namedtuple/dataclass returned by NLE)
    for key in (
        list(OBSERVATION_KEYS)
        + list(_ACTION_KEYS)
        + list(_CHAR_KEYS)
        + list(_COLOR_KEYS)
        + list(_BLSTATS_KEYS)
        + list(_MESSAGE_KEYS)
        + list(_GAMEID_KEYS)
    ):
        if hasattr(batch, key):
            out[key] = _as_array(getattr(batch, key))
    if not out:
        out["observations"] = _as_array(batch)
    return out


def _as_array(value: Any) -> Any:
    if value is None:
        return None
    if _HAS_TORCH and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if _HAS_NUMPY and isinstance(value, np.ndarray):
        return value
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value) if _HAS_NUMPY else list(value)
        except Exception:
            return list(value)
    return value


def extract_observations(batch: Mapping[str, Any], keys: Sequence[str] = OBSERVATION_KEYS) -> Dict[str, Any]:
    """Pull the model observation keys out of a dataset batch."""
    found: Dict[str, Any] = {}
    aliases = {
        "tty_chars": _CHAR_KEYS,
        "tty_colors": _COLOR_KEYS,
        "blstats": _BLSTATS_KEYS,
        "message": _MESSAGE_KEYS,
    }
    for key in keys:
        value = _resolve_attr(batch, aliases.get(key, (key,)))
        if value is not None:
            found[key] = value
    if not found and "observations" in batch:
        found["observations"] = batch["observations"]
    return found


def extract_actions(batch: Mapping[str, Any], num_actions: Optional[int] = None, as_tensor: bool = False) -> Any:
    """Extract integer Expert actions from a dataset batch.

    NLD-AA stores the *one-hot* action labels of the human player; if the
    batch contains one-hot vectors they are converted to indices.
    """
    actions = _resolve_attr(batch, _ACTION_KEYS)
    if actions is None:
        return None
    if _HAS_NUMPY:
        arr = np.asarray(actions)
        if arr.ndim >= 2 and arr.shape[-1] > 1 and num_actions is not None and arr.shape[-1] == num_actions:
            if np.isclose(arr.sum(axis=-1), 1.0).all():
                arr = arr.argmax(axis=-1)
        elif arr.ndim >= 2 and arr.shape[-1] > 1 and np.isclose(arr.sum(axis=-1), 1.0).all():
            arr = arr.argmax(axis=-1)
        actions = arr
    if as_tensor and _HAS_TORCH:
        return torch.as_tensor(actions, dtype=torch.long)
    return actions


def extract_game_ids(batch: Mapping[str, Any]) -> Any:
    """Episode/game identifiers (used to split BC buffer states by game)."""
    return _resolve_attr(batch, _GAMEID_KEYS)


def batch_size_of(batch: Mapping[str, Any]) -> int:
    """Best-effort leading dimension of a dataset batch."""
    for value in batch.values():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
    return 0


# ---------------------------------------------------------------------------
# High-level iteration helpers (used by train_nethack / fisher / BC)
# ---------------------------------------------------------------------------
def open_dataset(
    name: str = DATASET_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seq_length: int = DEFAULT_SEQ_LENGTH,
    num_batches: Optional[int] = None,
    shuffle: bool = True,
    loop_forever: bool = True,
    seed: Optional[int] = None,
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> TtyrecDataset:
    """Create a :class:`TtyrecDataset` iterator (Appendix B.1)."""
    return TtyrecDataset(
        name=name,
        batch_size=batch_size,
        seq_length=seq_length,
        num_batches=num_batches,
        shuffle=shuffle,
        loop_forever=loop_forever,
        seed=seed,
        data_dir=data_dir,
        device=device,
    )


def iterate_batches(
    name: str = DATASET_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_batches: Optional[int] = None,
    seq_length: int = DEFAULT_SEQ_LENGTH,
    shuffle: bool = True,
    loop_forever: bool = True,
    seed: Optional[int] = None,
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
    dataset: Any = None,
) -> Iterator[Dict[str, Any]]:
    """Yield normalised NLD-AA batches.

    ``num_batches=None`` iterates the dataset indefinitely when
    ``loop_forever`` is set, matching ``nle.dataset`` semantics.
    """
    if dataset is None:
        dataset = open_dataset(
            name=name,
            batch_size=batch_size,
            seq_length=seq_length,
            num_batches=num_batches,
            shuffle=shuffle,
            loop_forever=loop_forever,
            seed=seed,
            data_dir=data_dir,
            device=device,
        )
    count = 0
    iterator = iter(dataset)
    while True:
        if num_batches is not None and count >= num_batches:
            return
        try:
            yield next(iterator)
        except StopIteration:
            if loop_forever and num_batches is None:
                iterator = iter(dataset)
                continue
            return
        count += 1


def fisher_batches(
    num_batches: int = NUM_FISHER_BATCHES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    name: str = DATASET_NAME,
    seq_length: int = DEFAULT_SEQ_LENGTH,
    shuffle: bool = True,
    seed: Optional[int] = None,
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Iterator[Tuple[Dict[str, Any], Any]]:
    """Yield ``(observations, expert_actions)`` pairs for the Fisher estimate.

    The diagonal Fisher of the actor at ``theta_*`` is accumulated by summing
    squared gradients of the expert log-likelihood over ``10 000`` NLD-AA
    batches (Appendix C.1); this generator is exactly the batch source for
    :meth:`src.retention.fisher.FisherEstimator.compute`.
    """
    kwargs.pop("dataset", None)
    batches = iterate_batches(
        name=name,
        batch_size=batch_size,
        num_batches=num_batches,
        seq_length=seq_length,
        shuffle=shuffle,
        loop_forever=False,
        seed=seed,
        data_dir=data_dir,
        device=device,
    )
    for batch in batches:
        obs = extract_observations(batch)
        actions = extract_actions(batch)
        if actions is None:
            continue
        yield obs, actions


def expert_state_action_batches(
    num_batches: int = NUM_FISHER_BATCHES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    **kwargs: Any,
) -> Iterator[Tuple[Dict[str, Any], Any]]:
    """Alias of :func:`fisher_batches` (expert (state, action) stream)."""
    return fisher_batches(num_batches=num_batches, batch_size=batch_size, **kwargs)


def load_states(
    num_states: int = 10_000,
    batch_size: int = DEFAULT_BATCH_SIZE,
    name: str = DATASET_NAME,
    seed: Optional[int] = None,
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
    max_batches: Optional[int] = None,
    observation_keys: Sequence[str] = OBSERVATION_KEYS,
    as_tensor: bool = False,
) -> Dict[str, Any]:
    """Sample ``num_states`` pre-training states from NLD-AA.

    Returns a dict with ``observations`` (dict of arrays), ``actions``
    (expert actions, when available) and ``num_states``.  This is the state
    source for the behavioral-cloning buffer ``B_BC`` (Appendix C.2).
    """
    collected: Dict[str, List[Any]] = {key: [] for key in observation_keys}
    actions_collected: List[Any] = []
    total = 0
    needed_batches = max_batches
    if needed_batches is None:
        needed_batches = max(1, int(math.ceil(num_states / max(1, batch_size))))

    for index, batch in enumerate(
        iterate_batches(
            name=name,
            batch_size=batch_size,
            num_batches=needed_batches,
            shuffle=True,
            loop_forever=False,
            seed=seed,
            data_dir=data_dir,
            device=device,
        )
    ):
        obs = extract_observations(batch, keys=observation_keys)
        actions = extract_actions(batch)
        sizes = [batch_size_of(v) for v in obs.values() if hasattr(v, "shape")]
        n_items = max(sizes) if sizes else batch_size_of(batch)
        take = min(n_items, num_states - total)
        if take <= 0:
            break
        for key in observation_keys:
            value = obs.get(key)
            if value is None:
                continue
            value = _take(value, take)
            collected.setdefault(key, []).append(value)
        if actions is not None:
            actions_collected.append(_take(actions, take))
        total += take
        if total >= num_states:
            break

    observations = {key: _concat(values) for key, values in collected.items() if values}
    actions = _concat(actions_collected) if actions_collected else None
    if as_tensor and _HAS_TORCH:
        observations = {k: torch.as_tensor(v) for k, v in observations.items()}
        if actions is not None:
            actions = torch.as_tensor(actions)
    return {
        "observations": observations,
        "actions": actions,
        "num_states": int(total),
        "num_batches": index + 1 if "index" in dir() else None,
        "keys": tuple(observations.keys()),
    }


def sample_states(
    num_states: int = 10_000,
    batch_size: int = DEFAULT_BATCH_SIZE,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias of :func:`load_states` (used by ``train_nethack``)."""
    return load_states(num_states=num_states, batch_size=batch_size, **kwargs)


def _take(value: Any, n: int) -> Any:
    if value is None:
        return None
    try:
        return value[:n]
    except Exception:
        return value


def _concat(values: Sequence[Any]) -> Any:
    if not values:
        return None
    if _HAS_NUMPY:
        try:
            return np.concatenate([np.asarray(v) for v in values], axis=0)
        except Exception:
            pass
    if _HAS_TORCH and isinstance(values[0], torch.Tensor):
        return torch.cat(list(values), dim=0)
    return values


def count_expert_games(name: str = DATASET_NAME, data_dir: Optional[str] = None) -> Optional[int]:
    """Number of games in the registered dataset, when ``nle.dataset`` allows it."""
    mod = nle_dataset_module()
    if mod is None:
        return None
    try:
        import sqlite3

        path = database_path(name, data_dir=data_dir)
        if not os.path.exists(path):
            return None
        with sqlite3.connect(path) as conn:
            cursor = conn.execute("SELECT COUNT(DISTINCT gameid) FROM games")
            row = cursor.fetchone()
            return int(row[0]) if row else None
    except Exception:
        return None


def describe(name: str = DATASET_NAME, data_dir: Optional[str] = None) -> Dict[str, Any]:
    """Report dataset availability/inventory (used by CLIs and logs)."""
    data_dir = data_dir or default_data_dir()
    files = find_ttyrec_files(data_dir)
    db = database_path(name, data_dir=data_dir)
    return {
        "name": name,
        "nle_available": nle_available(),
        "data_dir": data_dir,
        "num_shards_expected": NUM_SHARDS,
        "num_games_expected": NUM_EPISODES,
        "batch_size": DEFAULT_BATCH_SIZE,
        "num_fisher_batches": NUM_FISHER_BATCHES,
        "database": db,
        "database_exists": os.path.exists(db),
        "registered": dataset_exists(name, data_dir=data_dir),
        "num_ttyrec_files": len(files),
        "num_games": count_expert_games(name, data_dir=data_dir),
    }


def export_bc_buffer(
    path: str,
    num_states: int = 10_000,
    batch_size: int = DEFAULT_BATCH_SIZE,
    **kwargs: Any,
) -> str:
    """Dump ``num_states`` NLD-AA states to ``.npz`` for the BC buffer."""
    data = load_states(num_states=num_states, batch_size=batch_size, **kwargs)
    if not _HAS_NUMPY:
        raise DatasetUnavailableError("numpy is required to export the BC buffer")
    payload: Dict[str, Any] = {}
    for key, value in data["observations"].items():
        payload[f"obs__{key}"] = np.asarray(value)
    if data.get("actions") is not None:
        payload["actions"] = np.asarray(data["actions"])
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NLD-AA dataset pipeline (Appendix B.1)")
    parser.add_argument("--data-dir", default=None, help="directory with shards / sqlite db")
    parser.add_argument("--name", default=DATASET_NAME, help="registered dataset name")
    parser.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-batches", type=int, default=None, help="batches to iterate")
    parser.add_argument("--pattern", default=None, help="force a shard file-name pattern")
    parser.add_argument("--download", action="store_true", help="download the shards")
    parser.add_argument("--build", action="store_true", help="build the sqlite database")
    parser.add_argument("--force", action="store_true", help="re-download existing shards")
    parser.add_argument("--describe", action="store_true", help="print the dataset inventory")
    parser.add_argument("--export-bc", default=None, help="write a .npz BC buffer to this path")
    parser.add_argument("--num-states", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    data_dir = args.data_dir or default_data_dir()

    if args.describe or not (args.download or args.build or args.export_bc):
        print(json.dumps(describe(args.name, data_dir=data_dir), indent=2))

    if args.download:
        shards = download_shards(
            dest=data_dir,
            num_shards=args.num_shards,
            pattern=args.pattern,
            force=args.force,
        )
        unpack_shards(shards, dest=data_dir)
        _log(f"downloaded {len(shards)} shards into {data_dir}")

    if args.build:
        db = build_dataset(data_dir=data_dir, name=args.name, num_shards=args.num_shards)
        _log(f"database ready: {db}")

    if args.export_bc:
        path = export_bc_buffer(
            args.export_bc,
            num_states=args.num_states,
            batch_size=args.batch_size,
            name=args.name,
            seed=args.seed,
            data_dir=data_dir,
        )
        _log(f"BC buffer exported to {path}")

    if args.num_batches:
        iterator = iterate_batches(
            name=args.name,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            data_dir=data_dir,
        )
        total = 0
        for batch in iterator:
            total += batch_size_of(batch)
        _log(f"iterated {args.num_batches} batches ({total} samples)")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
