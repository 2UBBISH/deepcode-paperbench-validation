"""C6: input/paper.md is byte-identical to run_trial.sh's shell concatenation; blacklist → denylist."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from apps.v2.agent.paper2code.config import RunPaths
from apps.v2.agent.paper2code.intake import IntakeError, compose_input, load_bundle, prepare_input, read_denylist


def _paper_dir(tmp_path: Path, *, addendum: str | None = "Use library X for baseline B.\n", blacklist: str | None = None) -> Path:
    root = tmp_path / "sapg"
    root.mkdir()
    (root / "paper.md").write_bytes(("# Title\n\n" + "line\n" * 8 + "ünïcode ✓\n").encode("utf-8"))
    if addendum is not None:
        (root / "addendum.md").write_text(addendum, encoding="utf-8")
    if blacklist is not None:
        (root / "blacklist.txt").write_text(blacklist, encoding="utf-8")
    (root / "rubric.json").write_text("{}")
    return root


def test_composition_matches_shell_byte_for_byte(tmp_path: Path) -> None:
    root = _paper_dir(tmp_path)
    shell = subprocess.run(
        ["bash", "-c", "{ cat \"$1/paper.md\"; printf '\\n\\n# Addendum\\n\\nClarifications provided with the paper by the benchmark authors (in scope; follow them):\\n\\n'; cat \"$1/addendum.md\"; }", "_", str(root)],
        capture_output=True, check=True,
    ).stdout
    data, included = compose_input(load_bundle(root))
    assert included
    assert data == shell


def test_no_or_empty_addendum_means_paper_only(tmp_path: Path) -> None:
    root = _paper_dir(tmp_path, addendum=None)
    data, included = compose_input(load_bundle(root))
    assert not included
    assert data == (root / "paper.md").read_bytes()
    (tmp_path / "b").mkdir()
    root2 = _paper_dir(tmp_path / "b", addendum="")
    data2, included2 = compose_input(load_bundle(root2))
    assert not included2
    assert data2 == (root2 / "paper.md").read_bytes()


def test_prepare_input_writes_file_and_records_sha(tmp_path: Path) -> None:
    root = _paper_dir(tmp_path, blacklist="# authors\nhttps://github.com/jayeshs999/sapg\n\n  github.com/other/impl  \n")
    paths = RunPaths(tmp_path / "run").ensure()
    record = prepare_input(load_bundle(root), paths)
    assert paths.paper_md.read_bytes() == compose_input(load_bundle(root))[0]
    assert record["denylist"] == ["https://github.com/jayeshs999/sapg", "github.com/other/impl"]
    assert record["rubric_present"] is True
    assert record["addendum_included"] is True
    assert len(record["paper_sha256"]) == 64
    assert record["paper_sha256"] != record["paper_md_sha256"]


def test_missing_or_short_paper_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(IntakeError):
        load_bundle(tmp_path / "nope")
    root = tmp_path / "short"
    root.mkdir()
    (root / "paper.md").write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(IntakeError, match="too short"):
        load_bundle(root)
    (root / "paper.md").write_text("x\n" * 10)
    assert read_denylist(load_bundle(root)) == ()
