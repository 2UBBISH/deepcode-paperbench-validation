"""The runtime spliced into every generated criteria file. Standard library only.

**Where the ruler runs.** The design document says the criteria file lives outside
the repository and is mounted read-only, so the agent cannot write to it. This
implementation goes one step further: the criteria file is never placed inside the
target container at all. pytest runs in a separate grader container on the same
remote Docker daemon, and every observation of the environment under test crosses
a `docker exec` bridge. Three things follow, and they are why the deviation is worth it:

* There is nothing in the target for the agent to tamper with -- not a mounted
  file, not a runner virtualenv, not a pytest installation. The grader receives
  the criterion through a separate read-only mount.
* The ruler needs no provisioning inside the container. Installing pytest there
  to run the criterion would have made point (1) of three-point falsification
  ambiguous: the bare container would no longer be bare.
* The environment under test still answers every question, because every check
  runs *through* it -- `loadable` deserialises the checkpoint with the target's
  own interpreter, which is the only interpreter whose opinion matters.

The module is spliced in verbatim rather than imported so the frozen artefact is
one self-contained file: hand it to a colleague and it runs. That also means its
source is inside `test_file_sha256`, so the ruler's own logic is frozen alongside
the assertions it evaluates.

Configuration arrives through the environment, set by `rsa.adjudicator`:

    RSA_EXEC       "docker" | "local"
    RSA_CONTAINER  container id            (docker)
    RSA_ROOT       host path of the repo   (local)
    RSA_WORKDIR    working directory for the target command
    RSA_ENV_JSON   json dict of env vars forwarded to the target command
    RSA_COMMAND    the frozen command
    RSA_TIMEOUT    seconds for one target run
    RSA_REPEATS    how many times to run the target (>= 2 for reproducibility)
    RSA_SNAPSHOT   json list of result files to copy aside after each run
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess

# --------------------------------------------------------------------------
# Bridge into the environment under test
# --------------------------------------------------------------------------

_EXEC = os.environ.get("RSA_EXEC", "local")
_CONTAINER = os.environ.get("RSA_CONTAINER", "")
_ROOT = os.environ.get("RSA_ROOT", "")
_WORKDIR = os.environ.get("RSA_WORKDIR", "/workspace/repo")
_ENV = json.loads(os.environ.get("RSA_ENV_JSON", "{}"))
_COMMAND = os.environ.get("RSA_COMMAND", "")
_TIMEOUT = int(os.environ.get("RSA_TIMEOUT", "1800"))
_REPEATS = int(os.environ.get("RSA_REPEATS", "1"))
_SNAPSHOT = json.loads(os.environ.get("RSA_SNAPSHOT", "[]"))

# Where each repeat's result files are copied. A second run at the same seed
# overwrites the first one's `results.json`, so without this the reproducibility
# check would compare a file against itself and pass unconditionally -- the exact
# shape of weak criterion the falsifier exists to catch.
_SNAP_DIR = "/tmp/rsa_snap"


class ProbeError(RuntimeError):
    """The bridge itself failed. Distinct from the environment failing a check.

    A dead container reads as every assertion failing at once, which looks
    identical to a catastrophically broken environment. It is not the same thing
    and must not be scored as one, so it raises instead of returning False.
    """


def sh(cmd: str, timeout: int = 120, workdir: str | None = None,
       env: dict | None = None) -> tuple[int, str]:
    """Run a shell command inside the environment under test."""
    wd = workdir or _WORKDIR
    merged = {**_ENV, **(env or {})}
    if _EXEC == "docker":
        flags: list[str] = []
        for k, v in merged.items():
            flags += ["-e", f"{k}={v}"]
        argv = ["docker", "exec", *flags, "-w", wd, _CONTAINER, "bash", "-lc", cmd]
    else:
        argv = ["bash", "-lc", f"cd {shlex.quote(wd)} && {cmd}"]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, **merged} if _EXEC == "local" else None)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    except OSError as e:  # docker missing, container gone
        raise ProbeError(f"exec bridge failed: {type(e).__name__}: {e}") from e
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _abspath(path: str) -> str:
    if path.startswith("/"):
        return path
    base = _WORKDIR if _EXEC == "docker" else (_ROOT or _WORKDIR)
    return f"{base.rstrip('/')}/{path}"


def exists(path: str) -> bool:
    return sh(f"test -e {shlex.quote(_abspath(path))}", timeout=60)[0] == 0


def size(path: str) -> int:
    """Bytes, or -1 when the path does not exist."""
    code, out = sh(
        f"stat -c %s {shlex.quote(_abspath(path))} 2>/dev/null || echo -1", timeout=60
    )
    for tok in reversed(out.split()):
        try:
            return int(tok)
        except ValueError:
            continue
    return -1


def sha256(path: str) -> str:
    code, out = sh(f"sha256sum {shlex.quote(_abspath(path))}", timeout=600)
    if code != 0:
        return ""
    return out.strip().split()[0] if out.strip() else ""


def read_text(path: str, limit: int = 4_000_000) -> str:
    code, out = sh(f"head -c {limit} {shlex.quote(_abspath(path))}", timeout=120)
    if code != 0:
        raise ProbeError(f"cannot read {path}: {out[:300]}")
    return out


def read_json(path: str):
    raw = read_text(path)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise AssertionError(f"{path} is not valid JSON: {e}") from None


# Deserialisation is delegated to the interpreter inside the container, because
# "the checkpoint loads" is a claim about the environment that was configured,
# not about the host. `python3` is used as the entry point and each loader falls
# back through the plausible module names rather than assuming one.
_LOADERS = {
    "json":    "import json;json.load(open(P,'rb'))",
    "torch":   "import torch;torch.load(P,map_location='cpu',weights_only=False)",
    "numpy":   "import numpy;numpy.load(P,allow_pickle=True)",
    "pickle":  "import pickle;pickle.load(open(P,'rb'))",
    "csv":     "import csv;list(csv.reader(open(P,newline='')))",
    "image":   "from PIL import Image;Image.open(P).load()",
    "yaml":    "import yaml;yaml.safe_load(open(P,'rb'))",
}


def loads_ok(path: str, loader: str) -> tuple[bool, str]:
    body = _LOADERS.get(loader)
    if body is None:
        raise ProbeError(f"unknown loader {loader!r}")
    prog = f"P={_abspath(path)!r}\n{body}\nprint('RSA_LOAD_OK')"
    code, out = sh("python3 - <<'RSA_EOF'\n" + prog + "\nRSA_EOF", timeout=900)
    return ("RSA_LOAD_OK" in out), out[-2000:]


# --------------------------------------------------------------------------
# The target run, executed once and shared by every test in the file
# --------------------------------------------------------------------------

_runs: list[dict] = []


def target_runs() -> list[dict]:
    """Execute the frozen command `RSA_REPEATS` times; cache for the whole session.

    Every assertion in a criteria file interrogates the same execution, so it has
    to happen exactly once per adjudication -- a file with thirty artefact checks
    must not launch thirty trainings. Repeats exist only for the reproducibility
    sanity check, which needs two runs at a fixed seed to have anything to compare.
    """
    if _runs:
        return _runs
    if not _COMMAND:
        raise ProbeError("RSA_COMMAND is empty: nothing to run")
    sh(f"rm -rf {_SNAP_DIR}", timeout=60)
    for i in range(max(1, _REPEATS)):
        code, out = sh(_COMMAND, timeout=_TIMEOUT)
        _runs.append({"exit_code": code, "output": out})
        _snapshot(i)
    return _runs


def _snapshot(run_index: int) -> None:
    """Copy this repeat's result files aside so a later repeat cannot hide it."""
    if not _SNAPSHOT:
        return
    dest = f"{_SNAP_DIR}/{run_index}"
    quoted = " ".join(shlex.quote(p) for p in _SNAPSHOT)
    # `|| true`: a result file the run failed to produce is a finding for the
    # artefact checks to report, not a reason to abort the snapshot of the rest.
    sh(f"mkdir -p {dest} && cp --parents -t {dest} {quoted} 2>/dev/null || true",
       timeout=300)


def _resolved(path: str, run_index: int) -> str:
    """The snapshot of `path` for a given repeat, falling back to the live file."""
    if run_index <= 0 and not _SNAPSHOT:
        return path
    snap = f"{_SNAP_DIR}/{run_index}/{path.lstrip('/')}"
    return snap if exists(snap) else path


def target() -> dict:
    return target_runs()[0]


# --------------------------------------------------------------------------
# Metric extraction
# --------------------------------------------------------------------------

def _dig(obj, jsonpath: list, path: str):
    for tok in jsonpath:
        try:
            obj = obj[tok]
        except (KeyError, IndexError, TypeError):
            rendered = "".join(f".{t}" if isinstance(t, str) else f"[{t}]" for t in jsonpath)
            raise AssertionError(f"{path}:{rendered} is absent (stopped at {tok!r})") from None
    return obj


def metric_from_json(path: str, jsonpath: list, run_index: int = 0) -> float:
    p = _resolved(path, run_index)
    return _as_float(_dig(read_json(p), jsonpath, p), f"{p}:{jsonpath}")


def metric_from_stdout(pattern: str, run_index: int = 0) -> float:
    """Last match wins: training logs print the same key every epoch."""
    out = target_runs()[run_index]["output"]
    found = re.findall(pattern, out)
    if not found:
        raise AssertionError(
            f"pattern /{pattern}/ never matched the run output "
            f"(exit_code={target_runs()[run_index]['exit_code']}); last 800 chars:\n"
            + out[-800:]
        )
    return _as_float(found[-1], f"/{pattern}/")


def _as_float(v, where: str) -> float:
    if isinstance(v, bool):
        raise AssertionError(f"{where} is a bool, not a number")
    try:
        return float(v)
    except (TypeError, ValueError):
        raise AssertionError(f"{where} is {v!r}, not a number") from None


def is_finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


def distinct_values(path: str, jsonpath: list, run_index: int = 0) -> int:
    """How many distinct values a prediction array holds.

    Guards the failure where a model predicts one class for everything and still
    scores well because the test split is imbalanced.
    """
    p = _resolved(path, run_index)
    obj = _dig(read_json(p), jsonpath, p)
    if not isinstance(obj, list):
        raise AssertionError(f"{p}: expected a list of predictions, got {type(obj).__name__}")
    return len({json.dumps(v, sort_keys=True) for v in obj})


def series(path: str, jsonpath: list, run_index: int = 0) -> list[float]:
    p = _resolved(path, run_index)
    obj = _dig(read_json(p), jsonpath, p)
    if not isinstance(obj, list) or not obj:
        raise AssertionError(f"{p}: expected a non-empty list, got {obj!r}"[:200])
    return [_as_float(v, p) for v in obj]


def series_from_stdout(pattern: str, run_index: int = 0) -> list[float]:
    out = target_runs()[run_index]["output"]
    found = re.findall(pattern, out)
    if not found:
        raise AssertionError(f"pattern /{pattern}/ never matched the run output")
    return [_as_float(v, f"/{pattern}/") for v in found]


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
