"""PLAN-3 item 4c: generate_code/ as a git history kept beside the code, bundle-able for the machine's git daemon."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from apps.v2.agent.paper2code.code_repo import GIT_DIR_NAME, CodeRepo, CodeRepoError
from apps.v2.agent_engine.experiment.git_daemon import local_bundle


def _code(tmp_path: Path) -> Path:
    code = tmp_path / "workspace" / "generate_code"
    (code / "pkg").mkdir(parents=True)
    (code / "main.py").write_text("print('hi')\n")
    (code / "pkg" / "__init__.py").write_text("")
    (code / "pkg" / "__pycache__").mkdir()
    (code / "pkg" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\x00")
    return code


def test_commit_keeps_the_product_directory_free_of_git_and_byte_code(tmp_path: Path) -> None:
    code = _code(tmp_path)
    repo = CodeRepo.for_run(tmp_path, code)
    assert not repo.initialised
    sha = repo.commit("round 0")
    assert len(sha) == 40
    assert repo.git_dir == tmp_path / GIT_DIR_NAME
    assert not (code / ".git").exists()
    assert repo.file_count() == 2  # main.py + pkg/__init__.py; the .pyc is excluded
    assert repo.commit("nothing changed") == sha  # no empty commits
    (code / "main.py").write_text("print('fixed')\n")
    assert repo.dirty()
    second = repo.commit("repair round 1")
    assert second != sha
    assert repo.head() == second
    assert not repo.dirty()


def test_bundle_from_the_git_dir_clones_the_work_tree(tmp_path: Path) -> None:
    code = _code(tmp_path)
    repo = CodeRepo.for_run(tmp_path, code)
    repo.commit("round 0")
    bundle = local_bundle(repo.git_dir, tmp_path / "out" / "repo.bundle")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(bundle), str(clone)], check=True, capture_output=True)
    assert (clone / "main.py").read_text() == "print('hi')\n"
    assert (clone / "pkg" / "__init__.py").is_file()
    assert not (clone / "pkg" / "__pycache__").exists()
    head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert head == repo.head()


def test_missing_code_directory_is_an_error(tmp_path: Path) -> None:
    repo = CodeRepo.for_run(tmp_path, tmp_path / "nowhere")
    with pytest.raises(CodeRepoError, match="does not exist"):
        repo.init()
    assert repo.head() is None


def test_restore_puts_the_tree_back_to_a_commit_and_removes_later_files(tmp_path: Path) -> None:
    code = tmp_path / "generate_code"
    code.mkdir()
    (code / "a.py").write_text("v1\n")
    repo = CodeRepo.for_run(tmp_path / "run", code)
    first = repo.commit("round 0")
    (code / "a.py").write_text("v2\n")
    (code / "b.py").write_text("added by a repair round\n")
    second = repo.commit("repair 1")
    assert first != second
    counts = repo.restore("pre_repair")
    assert counts == {"restored": 1, "removed": 1}
    assert (code / "a.py").read_text() == "v1\n"
    assert not (code / "b.py").exists()
    assert repo.head() == second  # the history is untouched; only the work tree moved
    assert repo.first_commit() == first
    with pytest.raises(CodeRepoError):
        repo.restore("nope")
