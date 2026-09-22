"""G1: is everything the run needs actually here? Checked before any configuring.

Section 5.4. On research repositories a large share of failures are datasets that
need a login, weights that were never published, a GPU that is too small, or a
driver that does not match the pinned framework -- and none of those is something
a setup agent can fix. Letting it discover that after three hours of installing is
pure waste, so every one of them is a deterministic check that runs first.

This is also the single biggest improvement to the quality of escalation. The
measured alternative is the agent deciding for itself that a problem is not the
environment's fault, which was wrong 59% of the time. A checklist is wrong 0% of
the time: either the file is there or it is not.

One rule that is easy to get wrong: **scale shrinks, requirements do not.** A
two-step G2 run still imports wandb at the top of the script, still opens the
label map, still wants the font the plotting stage uses. The asset list is
compiled from the whole pipeline, not from the shrunk portion of it.

Credentials are checked for **presence only**. Their values are never read into a
report -- live API keys sit in plaintext in the SetupX `.env` files on this host,
and an escalation card is something a user pastes into a chat window.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field, asdict

from .bridge import Bridge, BridgeError
from .criterion import Asset

OK = "ok"
MISSING = "missing"
UNKNOWN = "unknown"      # could not be determined -- never silently treated as ok


@dataclass
class AssetResult:
    name: str
    kind: str
    status: str
    detail: str = ""
    stage: str = ""
    why: str = ""

    @property
    def ok(self) -> bool:
        return self.status == OK


@dataclass
class AssetReport:
    results: list[AssetResult] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(r.status != OK for r in self.results)

    @property
    def missing(self) -> list[AssetResult]:
        return [r for r in self.results if r.status == MISSING]

    @property
    def unknown(self) -> list[AssetResult]:
        return [r for r in self.results if r.status == UNKNOWN]

    def to_dict(self) -> dict:
        return {"blocked": self.blocked, "results": [asdict(r) for r in self.results]}

    def summary(self) -> str:
        if not self.results:
            return "no assets declared"
        if not self.blocked:
            return f"all {len(self.results)} assets present"
        bits = [f"{len(self.missing)} missing"] if self.missing else []
        if self.unknown:
            bits.append(f"{len(self.unknown)} undeterminable")
        return "; ".join(bits)

    def facts(self) -> list[str]:
        return [f"{r.kind} {r.name}: {r.status.upper()} - {r.detail}"
                for r in self.results if r.status != OK]


class AssetGate:
    def __init__(self, bridge: Bridge, *, workdir: str = "/workspace/repo"):
        self.bridge = bridge
        self.workdir = workdir

    def check(self, assets: list[Asset]) -> AssetReport:
        rep = AssetReport()
        for a in assets:
            try:
                rep.results.append(self._one(a))
            except BridgeError as e:
                rep.results.append(AssetResult(
                    name=a.name, kind=a.kind, status=UNKNOWN, stage=a.stage, why=a.why,
                    detail=f"could not reach the environment: {e}"))
        return rep

    def _one(self, a: Asset) -> AssetResult:
        fn = {
            "path": self._path, "file_hash": self._hash, "env_var": self._env,
            "gpu": self._gpu, "cuda": self._cuda, "command": self._command,
        }[a.kind]
        r = fn(a)
        r.stage, r.why = a.stage, a.why
        return r

    # -- individual checks -------------------------------------------------

    def _path(self, a: Asset) -> AssetResult:
        q = shlex.quote(a.path)
        res = self.bridge.run(
            f"if [ -e {q} ]; then du -sb {q} 2>/dev/null | cut -f1; else echo MISSING; fi",
            timeout=120, workdir=self.workdir)
        out = res.output.strip().splitlines()[-1] if res.output.strip() else "MISSING"
        if out == "MISSING" or not res.ok:
            return AssetResult(a.name, a.kind, MISSING, f"{a.path} does not exist")
        # An empty directory is the shape a half-finished download leaves behind,
        # and it satisfies `-e`. It is not the dataset.
        if out.isdigit() and int(out) == 0:
            return AssetResult(a.name, a.kind, MISSING, f"{a.path} exists but is empty")
        return AssetResult(a.name, a.kind, OK, f"{a.path} present ({out} bytes)")

    def _hash(self, a: Asset) -> AssetResult:
        got = self.bridge.sha256(a.path)
        if not got:
            return AssetResult(a.name, a.kind, MISSING, f"{a.path} does not exist")
        if got != a.sha256:
            return AssetResult(a.name, a.kind, MISSING,
                               f"{a.path} sha256 {got[:16]}... != expected "
                               f"{a.sha256[:16]}...: the file is there but is not the "
                               "one this experiment was pinned to")
        return AssetResult(a.name, a.kind, OK, f"{a.path} matches its pinned hash")

    def _env(self, a: Asset) -> AssetResult:
        # Presence and length only. The value never enters a result.
        res = self.bridge.run(
            f'if [ -n "${a.env_var}" ]; then echo "SET ${{#{a.env_var}}}"; '
            f'else echo UNSET; fi', timeout=60, workdir=self.workdir)
        out = res.output.strip().splitlines()[-1] if res.output.strip() else "UNSET"
        if out.startswith("SET"):
            return AssetResult(a.name, a.kind, OK,
                               f"{a.env_var} is set ({out.split()[-1]} characters; "
                               "the value is deliberately not recorded)")
        return AssetResult(a.name, a.kind, MISSING, f"{a.env_var} is not set")

    def _gpu(self, a: Asset) -> AssetResult:
        res = self.bridge.run(
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits",
            timeout=120, workdir=self.workdir)
        out = res.output.strip()
        if not res.ok:
            # "No GPU" and "the driver is broken" are different problems with
            # different owners, and only one of them is about this machine having
            # the wrong hardware. Conflating them sends the user to buy a GPU they
            # already own. This host currently shows the second: NVML reports a
            # driver/library version mismatch with an RTX 3090 physically present.
            if "version mismatch" in out.lower() or "NVML" in out:
                return AssetResult(a.name, a.kind, UNKNOWN,
                                   "nvidia-smi cannot initialise NVML (driver/library "
                                   f"version mismatch): {out.strip()[:200]}. The GPU may "
                                   "well be present; the host's driver needs fixing "
                                   "before this can be answered.")
            return AssetResult(a.name, a.kind, MISSING,
                               f"nvidia-smi is unavailable: {out[:200]}")

        gpus = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[1].replace(".", "").isdigit():
                gpus.append((parts[0], float(parts[1]) / 1024.0))
        if not gpus:
            return AssetResult(a.name, a.kind, MISSING, "no GPU reported")

        best = max(gpus, key=lambda g: g[1])
        if a.min_vram_gb and best[1] < a.min_vram_gb:
            return AssetResult(a.name, a.kind, MISSING,
                               f"largest GPU is {best[0]} with {best[1]:.1f} GiB; "
                               f"{a.min_vram_gb} GiB required")
        if a.gpu_name_contains and not any(
                a.gpu_name_contains.lower() in g[0].lower() for g in gpus):
            return AssetResult(a.name, a.kind, MISSING,
                               f"no GPU matching {a.gpu_name_contains!r}; found "
                               + ", ".join(g[0] for g in gpus))
        return AssetResult(a.name, a.kind, OK,
                           f"{best[0]}, {best[1]:.1f} GiB")

    def _cuda(self, a: Asset) -> AssetResult:
        res = self.bridge.run("nvcc --version 2>/dev/null || nvidia-smi 2>&1",
                              timeout=120, workdir=self.workdir)
        m = re.search(r"(?:release |CUDA Version:\s*)(\d+)\.(\d+)", res.output)
        if not m:
            return AssetResult(a.name, a.kind, UNKNOWN,
                               "no CUDA version could be read from nvcc or nvidia-smi")
        got = (int(m.group(1)), int(m.group(2)))
        for bound, cmp_ in ((a.cuda_min, "min"), (a.cuda_max, "max")):
            if not bound:
                continue
            want = tuple(int(x) for x in bound.split(".")[:2])
            if cmp_ == "min" and got < want:
                return AssetResult(a.name, a.kind, MISSING,
                                   f"CUDA {got[0]}.{got[1]} is below the required {bound}")
            if cmp_ == "max" and got > want:
                return AssetResult(a.name, a.kind, MISSING,
                                   f"CUDA {got[0]}.{got[1]} is above the supported {bound}")
        return AssetResult(a.name, a.kind, OK, f"CUDA {got[0]}.{got[1]}")

    def _command(self, a: Asset) -> AssetResult:
        res = self.bridge.run(a.command, timeout=300, workdir=self.workdir)
        if res.ok:
            return AssetResult(a.name, a.kind, OK,
                               f"`{a.command}` succeeded: {res.output.strip()[:160]}")
        return AssetResult(a.name, a.kind, MISSING,
                           f"`{a.command}` exited {res.exit_code}: "
                           f"{res.output.strip()[-200:]}")
