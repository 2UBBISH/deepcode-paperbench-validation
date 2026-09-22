"""The Paper2Code line's footprint outside its own paths is exactly the registered one."""

import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "apps/v2/agent/paper2code/footprint.yaml"
MENTION = re.compile(r"(?<![a-z])paper2code", re.IGNORECASE)
KINDS = {"registration", "composition", "logic", "contract", "generated", "test", "tooling"}
# The root files the manifest may register although they sit outside scan_roots.
ROOT_FILES = ("pyproject.toml", "CONTEXT.md")


def manifest():
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def owned(rel: str) -> bool:
    return any("paper2code" in part.lower() for part in Path(rel).parts)


def mentions(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return 0
    return sum(1 for line in text.splitlines() if MENTION.search(line))


def scan(document) -> dict[str, int]:
    excluded_trees = tuple(document["excluded_trees"])
    excluded_suffixes = tuple(document["excluded_suffixes"])
    excluded_files = set(document["excluded_files"])
    found = {}
    for name in ROOT_FILES:
        count = mentions(ROOT / name)
        if count:
            found[name] = count
    for root in document["scan_roots"]:
        if not (ROOT / root).is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(ROOT / root):
            rel_dir = Path(dirpath).relative_to(ROOT).as_posix()
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__" and not f"{rel_dir}/{d}".startswith(excluded_trees))
            for filename in filenames:
                rel = f"{rel_dir}/{filename}"
                if rel in excluded_files or rel.endswith(excluded_suffixes) or owned(rel):
                    continue
                count = mentions(ROOT / rel)
                if count:
                    found[rel] = count
    return found


def test_manifest_is_well_formed():
    document = manifest()
    entries = document["registrations"]
    paths = [e["path"] for e in entries] + [f["path"] for f in document["foreign"]]
    assert len(set(paths)) == len(paths), "a file is registered twice"
    for entry in entries:
        assert entry["kind"] in KINDS, entry
        if "lines" in entry:
            assert isinstance(entry["lines"], int), entry
            assert entry["lines"] > 0, entry
    for entry in entries + list(document["foreign"]):
        assert (ROOT / entry["path"]).is_file(), f"registered file is gone: {entry['path']}"


def test_every_mention_outside_the_line_is_registered():
    document = manifest()
    found = scan(document)
    registered = {e["path"]: e for e in document["registrations"]}
    foreign = {f["path"] for f in document["foreign"]}
    unregistered = sorted(set(found) - set(registered) - foreign)
    assert not unregistered, (
        "shared files mention the paper2code line without a footprint entry; move the code into a path named"
        f" after the line or register the hook in footprint.yaml: {unregistered}")
    silent = sorted((set(registered) | foreign) - set(found))
    assert not silent, f"registered files no longer mention the line; drop them from footprint.yaml: {silent}"
    grown = {path: (found[path], entry["lines"]) for path, entry in registered.items() if "lines" in entry and found[path] > entry["lines"]}
    assert not grown, f"a bounded hook grew; move the new code into the line's own paths or raise `lines` on purpose (actual, allowed): {grown}"


def test_owned_paths_are_named_after_the_line():
    assert owned("apps/v2/agent/paper2code/driver.py")
    assert owned("apps/v2/agent_engine/paper2code/tools/git_command.py")
    assert owned("scripts/paper2code_canary.py")
    assert owned("tests/v2_paper2code/test_driver_offline.py")
    assert not owned("apps/v2/remote_compute/providers.py")
