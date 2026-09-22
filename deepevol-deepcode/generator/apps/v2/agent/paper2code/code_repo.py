"""The generated code as a git history, kept beside the code rather than inside it.

The experiment agent clones the code inside its container from a git daemon on the machine,
and main's ``git_daemon.serve_repo_on_machine`` feeds that daemon a ``git bundle --all`` of a
local repository. So ``generate_code/`` needs a git history: one commit per state the criterion is
judged against (round 0, then every repair round — the criterion pins a commit and the
adjudicator resets to it, PLAN-3 item 5).

The repository lives at ``<run>/code.git`` with ``generate_code/`` as its work tree
(``GIT_DIR`` / ``GIT_WORK_TREE``), never as a ``.git`` inside the product: the submission, the
ownership gate and the engine's own view of ``generate_code/`` stay exactly what they were.
``code.git/info/exclude`` keeps byte-code and caches out of the history. ``serve_repo_on_machine``
is pointed at ``code.git`` itself — git treats a directory that is a git dir as its own
repository, so ``git bundle`` runs there unchanged.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

GIT_DIR_NAME = "code.git"
EXCLUDES = ("__pycache__/", "*.pyc", "*.pyo", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/", ".ipynb_checkpoints/", ".DS_Store")
IDENTITY = ("-c", "user.name=paper2code", "-c", "user.email=paper2code@deepevol.local")


class CodeRepoError(RuntimeError):
    pass


@dataclass(slots=True)
class CodeRepo:
    work_tree: Path
    git_dir: Path

    @classmethod
    def for_run(cls, run_root: Path, code_dir: Path) -> "CodeRepo":
        return cls(work_tree=Path(code_dir), git_dir=Path(run_root) / GIT_DIR_NAME)

    # -- plumbing ------------------------------------------------------------------------------

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ, GIT_DIR=str(self.git_dir), GIT_WORK_TREE=str(self.work_tree))
        for stale in ("GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_NAMESPACE"):
            env.pop(stale, None)
        result = subprocess.run(["git", *IDENTITY, *args], cwd=str(self.work_tree), env=env, capture_output=True, text=True, check=False)
        if check and result.returncode != 0:
            raise CodeRepoError(f"git {' '.join(args)} failed (exit {result.returncode}): {(result.stderr or result.stdout).strip()[:400]}")
        return result

    @property
    def initialised(self) -> bool:
        return (self.git_dir / "HEAD").is_file()

    def init(self) -> None:
        """Create the repository once; harmless when it exists."""
        if not self.work_tree.is_dir():
            raise CodeRepoError(f"code directory {self.work_tree} does not exist")
        if self.initialised:
            return
        self.git_dir.parent.mkdir(parents=True, exist_ok=True)
        self._git("init", "-q", "-b", "main")
        (self.git_dir / "info").mkdir(exist_ok=True)
        (self.git_dir / "info" / "exclude").write_text("\n".join(EXCLUDES) + "\n", encoding="utf-8")

    # -- what the phases use -------------------------------------------------------------------

    def head(self) -> str | None:
        if not self.initialised:
            return None
        result = self._git("rev-parse", "--verify", "-q", "HEAD", check=False)
        return result.stdout.strip() or None

    def dirty(self) -> bool:
        self.init()
        return bool(self._git("status", "--porcelain", "--untracked-files=all").stdout.strip())

    def commit(self, message: str) -> str:
        """Record the work tree as it is now; returns HEAD (unchanged when nothing changed)."""
        self.init()
        self._git("add", "-A")
        if self.head() is not None and not self._git("status", "--porcelain").stdout.strip():
            return self.head() or ""
        self._git("commit", "-q", "--allow-empty-message", "-m", message)
        sha = self.head()
        if not sha:
            raise CodeRepoError("commit produced no HEAD")
        return sha

    def file_count(self) -> int:
        self.init()
        return len([line for line in self._git("ls-files").stdout.splitlines() if line.strip()])

    # -- snapshots (what the judge compares) ------------------------------------------------

    PRE_REPAIR = "pre_repair"

    def first_commit(self) -> str | None:
        """The first commit of the history: the code as the experiment agent first judged it (round 0)."""
        if not self.initialised or self.head() is None:
            return None
        result = self._git("rev-list", "--max-parents=0", "HEAD", check=False)
        first = result.stdout.strip().splitlines()
        return first[-1] if first else None

    def resolve_snapshot(self, name: str) -> str | None:
        """``pre_repair`` → the first commit; otherwise a commit-ish that must exist in the history."""
        if name == self.PRE_REPAIR:
            return self.first_commit()
        if not self.initialised:
            return None
        result = self._git("rev-parse", "--verify", "-q", f"{name}^{{commit}}", check=False)
        return result.stdout.strip() or None

    def restore(self, commit: str) -> dict[str, int]:
        """Put the work tree back to ``commit`` — every file of that commit as it was, and every file the history
        added later removed — without touching the history itself. This is how step 10 is repeated on the tree as
        first judged (the stage-9 tree = the first commit) instead of on the last repair round's edits. Returns
        the counts (``restored`` files written, ``removed`` later files deleted)."""
        self.init()
        sha = self.resolve_snapshot(commit)
        if sha is None:
            raise CodeRepoError(f"{commit!r} is not in the code history")
        wanted = {line for line in self._git("ls-tree", "-r", "--name-only", sha).stdout.splitlines() if line.strip()}
        tracked = {line for line in self._git("ls-files").stdout.splitlines() if line.strip()}
        removed = 0
        for rel in sorted(tracked - wanted):
            path = self.work_tree / rel
            if path.is_file():
                path.unlink()
                removed += 1
        self._git("checkout", "-q", sha, "--", ".")
        for rel in sorted(tracked - wanted):  # drop the deletions from the index too, so the tree and index agree
            self._git("rm", "-q", "--cached", "--ignore-unmatch", rel, check=False)
        return {"restored": len(wanted), "removed": removed}

    def export(self, commit: str, target: Path, *, excluded_names: frozenset[str] | set[str] = frozenset(), excluded_suffixes: tuple[str, ...] = ()) -> list[Path]:
        """Materialise ``commit`` under ``target`` (created); returns the files written."""
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(
            ["git", *IDENTITY, "archive", "--format=tar", commit],
            cwd=str(self.work_tree), env=dict(os.environ, GIT_DIR=str(self.git_dir), GIT_WORK_TREE=str(self.work_tree)),
            capture_output=True, check=False,
        )
        if archive.returncode != 0:
            raise CodeRepoError(f"git archive {commit[:12]} failed: {archive.stderr.decode(errors='replace')[:300]}")
        import io
        import tarfile

        written: list[Path] = []
        with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                if not member.isfile() or any(p in excluded_names for p in parts) or member.name.endswith(excluded_suffixes):
                    continue
                dest = target / member.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                dest.write_bytes(extracted.read())
                written.append(dest)
        return written


__all__ = ["EXCLUDES", "GIT_DIR_NAME", "CodeRepo", "CodeRepoError"]
