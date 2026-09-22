"""ADR 0004: Source pointers read back from the blueprint, section-level reading obligations, read receipts."""

from pathlib import Path

import pytest

from apps.v2.agent_engine.paper2code.workflows.paper_readback import PaperIndex
from apps.v2.agent_engine.paper2code.workflows.source_fidelity import (
    FidelityError,
    FidelitySession,
    audit,
    compile_manifest,
    exact_section,
    freeze_manifest,
    planned_files,
    relative_file,
    resolve_read_args,
)

PAPER = r"""# 1 Method

The total loss is
\[
L(\theta)=L_{on}(\theta)+\lambda L_{off}(\theta)
\]
with lambda equal to one.

# 2 Training

Five seeds, Adam, learning rate 3e-4.

# Results

The experiment uses five seeds.
"""

PLAN = r"""```yaml
complete_reproduction_plan:
  file_structure: |
    project/
      src/
        loss.py            # total loss
        train.py           # training loop
        __init__.py
      README.md
  implementation_components: |
    1) Total loss (project/src/loss.py)
       L = L_on + lambda * L_off with lambda from the paper.
       Source: §1 Method

    2) Training loop (project/src/train.py)
       Adam, five seeds; reports the mean.
       Source: §2, Results

    3) Package marker (project/src/__init__.py) and README.md — glue.
  validation_approach: minimal smoke
  environment_setup: python
  implementation_strategy: loss first
```
"""


def _task(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    (task / "generate_code" / "project" / "src").mkdir(parents=True)
    (task / "paper.md").write_text(PAPER, encoding="utf-8")
    (task / "initial_plan.txt").write_text(PLAN, encoding="utf-8")
    return task


def test_planned_files_reads_tree_drawings_and_plain_lists() -> None:
    assert planned_files(PLAN) == ["project/src/loss.py", "project/src/train.py", "project/src/__init__.py", "project/README.md"]
    plain = "```yaml\ncomplete_reproduction_plan:\n  file_structure: |\n    a/b.py\n    - ./c.yaml\n    docs/  # a directory\n    notes\n```"
    assert planned_files(plain) == ["a/b.py", "c.yaml"]


def test_manifest_binds_files_to_the_sections_their_paragraphs_point_at(tmp_path: Path) -> None:
    task = _task(tmp_path)
    manifest = freeze_manifest(task, PLAN)
    assert manifest["files"]["project/src/loss.py"]["sections"] == ["1"]
    assert manifest["files"]["project/src/train.py"]["sections"] == ["2", "Results"]
    assert manifest["glue"] == ["project/src/__init__.py", "project/README.md"]
    assert manifest["pointers"] == 2
    assert manifest["unmatched"] == []
    assert manifest["orphan_pointers"] == 0


def test_pointer_spellings_unmatched_pointers_and_orphan_paragraphs() -> None:
    index = PaperIndex.from_markdown("# 4 Method\n\nx\n\n## 4.1. Functional Reward Encoding\n\ny\n\n# A. Hyperparameters\n\nz\n")
    ids = {p: exact_section(index, p).section_id for p in ("4.1", "§4.1", "4.1. Functional Reward Encoding", "Functional Reward Encoding", "A. Hyperparameters", "Appendix A")}
    assert len(set(ids.values())) == 2
    with pytest.raises(FidelityError, match="SOURCE_SECTION_NOT_UNIQUE"):
        exact_section(index, "nowhere")
    plan = PLAN.replace("Source: §1 Method", "Source: §1 Method, §9.9").replace("    3) Package marker", "    3) A paragraph that points but names no file.\n       Source: §2\n\n    4) Package marker")
    manifest = compile_manifest(plan, PAPER.encode())
    assert manifest["files"]["project/src/loss.py"]["sections"] == ["1"]
    assert [u["pointer"] for u in manifest["unmatched"]] == ["§9.9"]
    assert manifest["orphan_pointers"] == 1
    assert manifest["pointers"] == 3


def test_file_paths_are_folded() -> None:
    assert relative_file("./project/src/loss.py") == "project/src/loss.py"
    assert relative_file("/workspace/generate_code/project//src/loss.py") == "project/src/loss.py"
    assert relative_file(" `generate_code/project/src/loss.py` ") == "project/src/loss.py"
    with pytest.raises(FidelityError, match="SOURCE_FILE_IS_DIRECTORY"):
        relative_file("project/scripts/")
    with pytest.raises(FidelityError, match=r"SOURCE_FILE_INVALID"):
        relative_file("../outside.py")


def test_write_needs_every_bound_section_read_in_an_earlier_turn(tmp_path: Path) -> None:
    task = _task(tmp_path)
    freeze_manifest(task, PLAN)
    session = FidelitySession(task, task / "generate_code")
    code = task / "generate_code"
    # a read for the file resolves to its next unread section; the same turn does not authorise the write
    args = {"file_path": "project/src/train.py", "section": ""}
    assert resolve_read_args(session, args) is None
    assert args["section"] == "2"
    session.read(**{k: args[k] for k in ("file_path", "section")})
    session.next_turn()
    with pytest.raises(FidelityError, match=r"SOURCE_READ_REQUIRED.*Results"):
        session.authorize("project/src/train.py")
    args = {"file_path": "project/src/train.py", "section": "§1 Method"}  # a section the file is not bound to → next unread
    assert resolve_read_args(session, args) is None
    assert args["section"] == "Results"
    session.read(file_path="project/src/train.py", section="Results")
    with pytest.raises(FidelityError, match="SOURCE_READ_REQUIRED"):
        session.authorize("project/src/train.py")  # the Results read is this turn
    session.next_turn()
    (code / "project" / "src" / "train.py").write_text("train = 1\n", encoding="utf-8")
    receipts = session.authorize("project/src/train.py")
    assert len(receipts) == 2
    session.written("project/src/train.py", receipts)
    # glue needs nothing; a file outside the plan is allowed and unbound
    assert "SOURCE_GLUE_FILE" in resolve_read_args(session, {"file_path": "project/README.md"})
    assert session.authorize("project/README.md") == []
    (code / "project" / "README.md").write_text("docs\n", encoding="utf-8")
    session.written("project/README.md", [])
    assert session.authorize("project/src/extra.py") == []
    # loss.py: read, next turn, write
    session.read(file_path="project/src/loss.py")
    session.next_turn()
    (code / "project" / "src" / "loss.py").write_text("L = 1\n", encoding="utf-8")
    session.written("project/src/loss.py", session.authorize("project/src/loss.py"))
    (code / "project" / "src" / "__init__.py").write_text("", encoding="utf-8")
    session.written("project/src/__init__.py", [])
    report = audit(task, code)
    assert report["passed"] is True
    assert report["read_receipts"] == 3
    assert report["paper_files"] == 2
    assert report["glue_files"] == 2


def test_audit_records_violations_instead_of_raising(tmp_path: Path) -> None:
    task = _task(tmp_path)
    freeze_manifest(task, PLAN)
    session = FidelitySession(task, task / "generate_code")
    code = task / "generate_code"
    # write loss.py with no read at all (the tool would have refused; the trace shows it anyway)
    (code / "project" / "src" / "loss.py").write_text("L = 1\n", encoding="utf-8")
    session.written("project/src/loss.py", [])
    (code / "project" / "src" / "stray.py").write_text("x\n", encoding="utf-8")
    (code / "project" / "src" / "pkg").mkdir()
    (code / "project" / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")  # package markers are never unplanned
    (code / "README.md").write_text("moved\n", encoding="utf-8")  # a planned file under another directory is not missing
    report = audit(task, code)
    assert report["passed"] is False
    assert any("bound section 1 not fully read" in v for v in report["violations"])
    assert report["unplanned"] == ["project/src/stray.py"]
    assert "project/src/train.py" in report["missing"]
    assert "project/README.md" not in report["missing"]


def test_manifest_frozen_before_implementation_cannot_change_afterwards(tmp_path: Path) -> None:
    task = _task(tmp_path)
    freeze_manifest(task, PLAN)
    session = FidelitySession(task, task / "generate_code")
    session.read(file_path="project/src/loss.py")
    with pytest.raises(FidelityError, match="SOURCE_PLAN_CHANGED_AFTER_IMPLEMENTATION"):
        freeze_manifest(task, PLAN.replace("Source: §1 Method", "Source: §2"))


def test_pointers_are_parsed_the_way_planners_write_them() -> None:
    # fre-t17 / rice-t17 first plans: 20 and 25 pointers unmatched for parentheticals, "Table 3" suffixes, quoted titles
    from apps.v2.agent_engine.paper2code.workflows.source_fidelity import resolve_pointer, split_pointers

    refs = split_pointers('§4.1 (Practical Implementation), Appendix A Table 3, Addendum "Additional Details on GC-BC / GC-IQL / OPAL", Algorithm 1')
    assert refs == ["Additional Details on GC-BC / GC-IQL / OPAL", "§4.1 (Practical Implementation)", "Appendix A Table 3", "Addendum", "Algorithm 1"]
    index = PaperIndex.from_markdown(
        "# 4 Method\n\nx\n\n## 4.1 Encoding\n\nPractical Implementation. Algorithm 1 is used here.\n\n# A Hyperparameters\n\nTable 3 lists them.\n\n# Addendum\n\nnotes\n\n# Additional Details on GC-BC / GC-IQL / OPAL\n\nmore\n"
    )
    assert resolve_pointer(index, "§4.1 (Practical Implementation") == "4.1"
    assert resolve_pointer(index, "§4.1 (Equation 6)") == "4.1"
    assert resolve_pointer(index, "Appendix A Table 3") == "A"
    assert resolve_pointer(index, "Additional Details on GC-BC / GC-IQL / OPAL") == "Additional Details on GC-BC / GC-IQL / OPAL"
    assert resolve_pointer(index, "Algorithm 1") == "4.1"
    assert resolve_pointer(index, "Addendum") == "Addendum"
    assert resolve_pointer(index, "§9.9 (nowhere)") is None


def test_more_pointer_spellings_from_the_second_batch() -> None:
    # sapg / bbox / lbcs / lca plans: number + words, Eq.(6, ranges, slashes
    from apps.v2.agent_engine.paper2code.workflows.source_fidelity import resolve_pointers, split_pointers

    index = PaperIndex.from_markdown(
        "# 2 Objective\n\nEq. (1) here.\n\n# 3 Method\n\n## 3.2 Details\n\nDefinition 1 and Eq. (14) live here.\n\n## 3.4 Initialization\n\nz\n\n# 6 Experiments\n\n## 6.1 A\n\na\n\n## 6.2 B\n\nb\n\n## 6.3 C\n\nc\n\n## 6.4 D\n\nd\n"
    )
    assert resolve_pointers(index, "§3.4 Initialization") == ["3.4"]
    assert resolve_pointers(index, "§2 Objective formulations") == ["2"]
    assert resolve_pointers(index, "§3.2 Definition 1") == ["3.2"]
    assert resolve_pointers(index, "Eq.(14) threshold update") == ["3.2"]
    assert resolve_pointers(index, "Eq.(1") == ["2"]
    assert resolve_pointers(index, "§6.1-§6.4") == ["6.1", "6.2", "6.3", "6.4"]
    assert split_pointers("Appendix D.2/Table 7 (SVHN), §2") == ["Appendix D.2", "Table 7 (SVHN)", "§2"]
    assert resolve_pointers(index, "momentum=0.9") == []


def test_directory_paragraphs_and_example_headings() -> None:
    # lbcs: "8) Models (models/) … Source: §5.1" binds every planned file under models/; bbox: a "#### 1" inside a
    # worked example must not make "§1" ambiguous
    from apps.v2.agent_engine.paper2code.workflows.source_fidelity import _files_in

    files = ["proj/models/cnn.py", "proj/models/resnet.py", "proj/data/loader.py", "proj/main.py"]
    assert _files_in("8) Models (models/) implement the CNNs.", files) == ["proj/models/cnn.py", "proj/models/resnet.py"]
    assert _files_in("entry proj/main.py and data/", files) == ["proj/data/loader.py", "proj/main.py"]
    index = PaperIndex.from_markdown("# 1 Introduction\n\nx\n\n#### 1\n\nan example line\n\n# 2 Method\n\ny\n")
    assert exact_section(index, "§1").section_id == "1"


def test_a_multi_page_section_must_be_read_whole(tmp_path: Path) -> None:
    """owner 09-21: a pointer names a section, so the section is read — every page — before the file is written;
    read_paper(file_path) walks the pages and says what is left."""
    long_paper = PAPER.replace("Five seeds, Adam, learning rate 3e-4.", "Five seeds, Adam, learning rate 3e-4. " + ("Details of the training loop. " * 260))
    task = tmp_path / "task"
    (task / "generate_code" / "project" / "src").mkdir(parents=True)
    (task / "paper.md").write_text(long_paper, encoding="utf-8")
    (task / "initial_plan.txt").write_text(PLAN, encoding="utf-8")
    freeze_manifest(task, PLAN)
    session = FidelitySession(task, task / "generate_code")
    assert session.pages("2") == 3
    assert session.unread_pages("project/src/train.py") == [("2", 1), ("2", 2), ("2", 3), ("Results", 1)]
    args = {"file_path": "project/src/train.py", "section": "", "part": 1}
    assert resolve_read_args(session, args) is None
    assert (args["section"], args["part"]) == ("2", 1)
    text = session.read(file_path="project/src/train.py", section="2", part=1)
    assert '"still_unread_for_file": ["§2 p2", "§2 p3", "§Results p1"]' in text
    session.next_turn()
    with pytest.raises(FidelityError, match=r"SOURCE_READ_REQUIRED.*§2 p2"):
        session.authorize("project/src/train.py")
    for _ in range(3):
        args = {"file_path": "project/src/train.py", "section": "", "part": 1}
        resolve_read_args(session, args)
        session.read(file_path=args["file_path"], section=args["section"], part=args["part"])
    assert "none — every bound page" in session.read(file_path="project/src/train.py", section="Results", part=1) or session.unread_pages("project/src/train.py") == []
    session.next_turn()
    (task / "generate_code" / "project" / "src" / "train.py").write_text("t = 1\n", encoding="utf-8")
    assert len(session.authorize("project/src/train.py")) >= 4
