"""The paper as PaperBench hands it over → the engine's single Markdown input.

A paper directory carries ``paper.pdf``, ``paper.md``, ``addendum.md``,
``blacklist.txt``, ``rubric.json``, ``config.yaml`` and ``assets/``. This
line reads three of them: ``paper.md`` is required (no PDF conversion here;
PaperBench's own Markdown is the caliber), ``addendum.md`` is appended as a
``# Addendum`` section byte-for-byte as the validation repo's
``run_trial.sh`` did, ``blacklist.txt`` becomes the run's denylist. The
rubric and the config are neither copied nor read: the ``criteria`` phase
only records whether a rubric exists.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apps.v2.agent.paper2code.config import RunPaths
from apps.v2.agent.paper2code.tools.registry import parse_denylist

ADDENDUM_HEADER = (
    b"\n\n# Addendum\n\n"
    b"Clarifications provided with the paper by the benchmark authors (in scope; follow them):\n\n"
)
MIN_PAPER_LINES = 5


class IntakeError(ValueError):
    """The paper directory is not usable as the benchmark hands it over."""


@dataclass(frozen=True, slots=True)
class PaperBundle:
    paper_dir: Path
    paper_md: Path
    addendum_md: Path | None
    blacklist_txt: Path | None
    rubric_json: Path | None
    config_yaml: Path | None

    @property
    def has_rubric(self) -> bool:
        return self.rubric_json is not None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(paper_dir: str | Path) -> PaperBundle:
    root = Path(paper_dir).expanduser().resolve()
    if not root.is_dir():
        raise IntakeError(f"paper directory does not exist: {root}")
    paper_md = root / "paper.md"
    if not paper_md.is_file():
        raise IntakeError(f"{root} has no paper.md (this line takes PaperBench's Markdown, not the PDF)")
    if len(paper_md.read_text(encoding="utf-8", errors="replace").splitlines()) <= MIN_PAPER_LINES:
        raise IntakeError(f"{paper_md} is too short to be a paper (LFS pointer not hydrated?)")

    def optional(name: str) -> Path | None:
        candidate = root / name
        return candidate if candidate.is_file() else None

    return PaperBundle(
        paper_dir=root,
        paper_md=paper_md,
        addendum_md=optional("addendum.md"),
        blacklist_txt=optional("blacklist.txt"),
        rubric_json=optional("rubric.json"),
        config_yaml=optional("config.yaml"),
    )


def compose_input(bundle: PaperBundle) -> tuple[bytes, bool]:
    """``paper.md`` bytes, plus the addendum section when ``addendum.md`` exists and is non-empty.

    Byte-identical to run_trial.sh's
    ``{ cat paper.md; printf '\\n\\n# Addendum\\n\\n...:\\n\\n'; cat addendum.md; }``.
    """
    body = bundle.paper_md.read_bytes()
    if bundle.addendum_md is not None and bundle.addendum_md.stat().st_size > 0:
        return body + ADDENDUM_HEADER + bundle.addendum_md.read_bytes(), True
    return body, False


def read_denylist(bundle: PaperBundle) -> tuple[str, ...]:
    if bundle.blacklist_txt is None:
        return ()
    return parse_denylist(bundle.blacklist_txt.read_text(encoding="utf-8", errors="replace"))


def prepare_input(bundle: PaperBundle, paths: RunPaths) -> dict[str, Any]:
    """Write ``input/paper.md`` and return what ``run.json`` records about it."""
    paths.input_dir.mkdir(parents=True, exist_ok=True)
    data, with_addendum = compose_input(bundle)
    paths.paper_md.write_bytes(data)
    return {
        "paper_dir": str(bundle.paper_dir),
        "paper_sha256": hashlib.sha256(data).hexdigest(),
        "paper_md_sha256": sha256_file(bundle.paper_md),
        "addendum_included": with_addendum,
        "denylist": list(read_denylist(bundle)),
        "rubric_present": bundle.has_rubric,
        "input_bytes": len(data),
    }


__all__ = [
    "ADDENDUM_HEADER",
    "IntakeError",
    "PaperBundle",
    "compose_input",
    "load_bundle",
    "prepare_input",
    "read_denylist",
    "sha256_file",
]
