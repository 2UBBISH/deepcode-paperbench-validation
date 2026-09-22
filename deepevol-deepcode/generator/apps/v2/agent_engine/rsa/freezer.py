"""Freeze a criterion ladder: render, measure the expected set, hash, store.

After this point the criteria are read-only. Not "we agree not to change them" --
read-only, with a hash the Adjudicator re-checks before every verdict, and an
amendment path that requires an explicit human approval and leaves a record.

The reason is narrow and measured. A criterion authored *after* the agent has
been struggling for an hour is a criterion authored to be reachable, and the
system has no way to notice: on the SWE-smith 128, 62/62 of gemma's failures and
33/34 of deepseek's had already declared themselves complete. The frozen artefact
is what makes "it passed" a claim about the environment rather than about the
system's own patience.

Two things get measured here rather than assumed:

* **`expected` comes from a real `pytest --collect-only`,** not from the renderer's
  belief about what it emitted. The renderer returns names too, and they are
  cross-checked -- a disagreement means the renderer is broken and freezing would
  bake a wrong expectation into everything downstream.
* **An empty `expected` set is refused.** It is satisfied by every possible run,
  including one that does nothing, and it fails silently rather than loudly.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .criterion import Criterion, Ladder, Rung, canonical_json
from .render import criteria_filename, render, snapshot_paths


class FreezeError(RuntimeError):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class FrozenCriterion:
    """Everything the Adjudicator needs, and nothing it could be talked out of."""

    rung: Rung
    criterion: Criterion
    path: Path                 # the criteria .py on the host
    test_file_sha256: str
    expected: list[str]
    frozen_at: float
    criterion_hash: str

    @property
    def basename(self) -> str:
        return self.path.name

    def check_integrity(self) -> None:
        """The file on disk must still be the file that was frozen."""
        if not self.path.exists():
            raise FreezeError(f"criteria file vanished: {self.path}")
        got = sha256_file(self.path)
        if got != self.test_file_sha256:
            raise FreezeError(
                f"criteria file {self.path.name} was modified after freezing "
                f"(sha256 {got[:12]} != frozen {self.test_file_sha256[:12]}). "
                "Refusing to adjudicate against an unfrozen ruler."
            )


@dataclass
class FrozenLadder:
    ladder_id: str
    ladder: Ladder
    root: Path
    rungs: dict[Rung, FrozenCriterion] = field(default_factory=dict)

    def check_integrity(self) -> None:
        for fc in self.rungs.values():
            fc.check_integrity()

    def get(self, rung: Rung) -> FrozenCriterion:
        try:
            return self.rungs[rung]
        except KeyError:
            raise FreezeError(f"no frozen criterion for rung {rung.value}") from None


class Freezer:
    """Content-addressed store of frozen criterion ladders."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- ids -------------------------------------------------------------

    @staticmethod
    def ladder_id(ladder: Ladder) -> str:
        """Identity is (repo, commit, goal). The rungs may be amended; this may not.

        `\\x00` as the separator rather than a printable character, matching
        `harness/verifier_hint.py:59`: a goal containing the separator must not be
        able to collide with a different (repo, commit) pair.
        """
        key = f"{ladder.repo_url}\x00{ladder.commit}\x00{ladder.goal}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def dir_for(self, ladder_id: str) -> Path:
        return self.root / ladder_id

    # -- freezing --------------------------------------------------------

    def freeze(self, ladder: Ladder, *, overwrite: bool = False,
               collector: Callable[[Path, int], list[str]] | None = None) -> FrozenLadder:
        lid = self.ladder_id(ladder)
        d = self.dir_for(lid)
        if d.exists() and not overwrite:
            existing = self.load(lid)
            if existing.ladder.canonical_json() == ladder.canonical_json():
                return existing
            raise FreezeError(
                f"ladder {lid} is already frozen with different content. "
                "Use Freezer.amend() with an explicit approval; freezing again "
                "would erase the record of what the earlier run was judged against."
            )
        d.mkdir(parents=True, exist_ok=True)

        frozen: dict[Rung, FrozenCriterion] = {}
        for c in ladder.ordered():
            frozen[c.rung] = self._freeze_one(c, d, collector=collector)

        fl = FrozenLadder(ladder_id=lid, ladder=ladder, root=d, rungs=frozen)
        self._write_manifest(fl)
        self._append_history(d, {
            "event": "freeze",
            "at": time.time(),
            "by": os.environ.get("USER", "?"),
            "ladder_hash": sha256_text(ladder.canonical_json()),
            "rungs": {r.value: fc.test_file_sha256 for r, fc in frozen.items()},
        })
        return fl

    def _freeze_one(self, c: Criterion, d: Path, *,
                    collector: Callable[[Path, int], list[str]] | None = None) -> FrozenCriterion:
        # Free code is linted here rather than at compile time, because here is the
        # only place it cannot be skipped: everything that reaches the Adjudicator
        # came through freeze().
        if c.free_pytest:
            from .linter import check as lint_check
            lint_check(c.free_pytest)
        source, predicted = render(c)
        path = d / criteria_filename(c)
        path.write_text(source, encoding="utf-8")

        collected = collector(path, 120) if collector is not None else collect_test_ids(path)
        if not collected:
            raise FreezeError(
                f"{path.name}: pytest collected no tests. An empty expected set is "
                "satisfied by every possible run, including one that does nothing."
            )

        # The renderer's own account of what it emitted, checked against what
        # pytest actually sees. They disagree when a fragment fails to render, or
        # when free code declares a test the file does not define.
        want = {f"{path.name}::{n}" for n in predicted}
        got = set(collected)
        if want != got:
            raise FreezeError(
                f"{path.name}: renderer predicted {len(want)} tests, pytest collected "
                f"{len(got)}.\n  only predicted: {sorted(want - got)[:10]}\n"
                f"  only collected: {sorted(got - want)[:10]}"
            )

        # Written once, hashed once. Everything downstream compares against this.
        c.expected = sorted(collected)
        c.test_file_sha256 = sha256_file(path)
        return FrozenCriterion(
            rung=c.rung,
            criterion=c,
            path=path,
            test_file_sha256=c.test_file_sha256,
            expected=list(c.expected),
            frozen_at=time.time(),
            criterion_hash=c.content_hash(),
        )

    # -- loading ---------------------------------------------------------

    def load(self, ladder_id: str) -> FrozenLadder:
        d = self.dir_for(ladder_id)
        mf = d / "frozen.json"
        if not mf.exists():
            raise FreezeError(f"no frozen ladder at {d}")
        blob = json.loads(mf.read_text(encoding="utf-8"))
        ladder = Ladder.from_dict(blob["ladder"])
        rungs: dict[Rung, FrozenCriterion] = {}
        for rv, row in blob["rungs"].items():
            r = Rung(rv)
            rungs[r] = FrozenCriterion(
                rung=r,
                criterion=ladder.rungs[r],
                path=d / row["file"],
                test_file_sha256=row["test_file_sha256"],
                expected=row["expected"],
                frozen_at=row["frozen_at"],
                criterion_hash=row["criterion_hash"],
            )
        return FrozenLadder(ladder_id=ladder_id, ladder=ladder, root=d, rungs=rungs)

    def _write_manifest(self, fl: FrozenLadder) -> None:
        blob = {
            "ladder_id": fl.ladder_id,
            "ladder": fl.ladder.to_dict(),
            "rungs": {
                r.value: {
                    "file": fc.basename,
                    "test_file_sha256": fc.test_file_sha256,
                    "criterion_hash": fc.criterion_hash,
                    "expected": fc.expected,
                    "frozen_at": fc.frozen_at,
                    "snapshot": snapshot_paths(fc.criterion),
                }
                for r, fc in sorted(fl.rungs.items(), key=lambda kv: kv[0].index)
            },
        }
        (fl.root / "frozen.json").write_text(
            json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # -- amendment -------------------------------------------------------

    def amend(self, ladder_id: str, rung: Rung, new_criterion: Criterion, *,
              approved_by: str, reason: str) -> FrozenLadder:
        """Replace one rung. Requires a named approver and a reason, both recorded.

        Amendment exists because falsification can prove a criterion wrong -- that
        is the whole point of the Falsifier. What it must never become is the
        system quietly relaxing a criterion it cannot meet, so the approver and the
        reason are mandatory arguments rather than optional metadata.
        """
        if not approved_by or not reason:
            raise FreezeError("amend() requires both approved_by and reason")
        fl = self.load(ladder_id)
        old = fl.rungs.get(rung)
        fl.ladder.rungs[rung] = new_criterion
        fl.rungs[rung] = self._freeze_one(new_criterion, fl.root)
        self._write_manifest(fl)
        self._append_history(fl.root, {
            "event": "amend",
            "at": time.time(),
            "rung": rung.value,
            "approved_by": approved_by,
            "reason": reason,
            "from_hash": old.criterion_hash if old else None,
            "to_hash": fl.rungs[rung].criterion_hash,
            "from_expected_n": len(old.expected) if old else 0,
            "to_expected_n": len(fl.rungs[rung].expected),
        })
        return fl

    def history(self, ladder_id: str) -> list[dict]:
        p = self.dir_for(ladder_id) / "history.jsonl"
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    @staticmethod
    def _append_history(d: Path, record: dict) -> None:
        with (d / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(canonical_json(record) + "\n")


def collect_test_ids(path: Path, timeout: int = 120) -> list[str]:
    """Ask pytest what is in the file. Ids are relative to the criteria directory.

    Invoked with `cwd` set to the file's directory and only its basename on the
    command line, so ids read `criteria_G2.py::test_x` no matter where the store
    lives. That is what makes `expected` portable between the machine that froze
    the criterion and the machine that adjudicates it -- the mismatch that cost
    `gunicorn` 98 of its 245 expected tests came from exactly this.
    """
    argv = [sys.executable, "-m", "pytest", "--collect-only", "-q",
            "-p", "no:cacheprovider", path.name]
    try:
        p = subprocess.run(argv, cwd=path.parent, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        raise FreezeError(f"collecting {path.name} timed out after {timeout}s") from None
    except OSError as e:
        raise FreezeError(f"cannot run pytest: {e}") from e

    out = (p.stdout or "") + (p.stderr or "")
    ids = [l.strip() for l in out.splitlines()
           if "::" in l and l.strip().startswith(path.name)]
    if not ids and p.returncode != 0:
        raise FreezeError(f"pytest could not collect {path.name}:\n{out[-2000:]}")
    # Deduplicate while keeping collection order stable for readable diffs.
    seen: set[str] = set()
    return [i for i in ids if not (i in seen or seen.add(i))]
