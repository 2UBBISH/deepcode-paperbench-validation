"""Source pointers, reading obligations and read receipts (ADR 0004; supersedes ADR 0003).

The blueprint's Section 2 already says, per component, which files it plans and where in the paper the component
comes from (``Source: §4.1``). The host reads that back mechanically — no second model call, no JSON binding, no
obligation ids, no anchors (ADR 0003's machinery, retired 2026-09-20 night: every failure in the 50-paper planner
batch was in that layer, and it added a judgement — "which obligation does this file implement" — that nothing could
verify anyway). What the host keeps:

* **Source pointer** — ``Source: §x.y`` in a Section 2 paragraph, resolved against the paper's headings.
* **Reading obligation** — a planned file is bound to every pointer of every paragraph that names it; before the
  coding agent writes the file it must have read each bound section (any page, an earlier model turn). A file no
  paragraph points at is *glue* and needs no read. A pointer that names no heading is *unmatched*: recorded, no
  obligation.
* **Read receipt** — the host's own record of one ``read_paper`` page (section, part, bytes); the write of a file
  cites the receipts that authorised it. The trace is what the end-of-implement audit replays; the audit is a
  record, not a gate.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .paper_readback import DEFAULT_PART_CHARS, PaperIndex
from .planning_runtime import extract_yaml_candidate

MANIFEST = "source_manifest.json"
TRACE = "source_trace.json"
VERSION = "paper-source.v2"

_SOURCE_LINE = re.compile(r"Source:\s*(?P<refs>[^\n]+)")
_SOURCE_SPLIT = re.compile(r"\s*(?:,|;|/| and |&)\s*")
_NOT_SPECIFIED = re.compile(r"not\s+specified|unspecified|n/?a\b|none", re.I)
_PATH_TOKEN = re.compile(r"[\w][\w./-]*\.[A-Za-z0-9]{1,12}")
_DIR_TOKEN = re.compile(r"(?<![\w./-])((?:[\w.-]+/)+)(?=[\s)\]`,;:]|$)")
_FILE_LINE = re.compile(r"^[\s│├└─|`*-]*(?P<path>[\w][\w./-]*\.[A-Za-z0-9]{1,12})(?=\s|$|#)")
_CODE_EXT = {
    "py", "pyx", "pyi", "ipynb", "yaml", "yml", "json", "toml", "cfg", "ini", "txt", "md", "rst", "sh", "bash", "csv",
    "tsv", "lock", "in", "env", "conf", "js", "ts", "jsx", "tsx", "cu", "c", "cc", "cpp", "h", "hpp", "java", "kt",
    "r", "jl", "m", "lua", "go", "rs", "proto", "cmake", "mk", "dockerfile", "gitignore", "cff", "bib",
}


class FidelityError(ValueError):
    pass


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def relative_file(value: str) -> str:
    """A planned file's path relative to generate_code. Spellings the planner uses are folded (``./x``, ``x//y``,
    ``generate_code/x``, ``/workspace/generate_code/x``, surrounding whitespace or backticks)."""
    if not isinstance(value, str) or "\x00" in value:
        raise FidelityError(f"SOURCE_FILE_INVALID: {value!r}")
    text = value.strip().strip("`").replace("\\", "/")
    text = re.sub(r"^(?:\./)+", "", text)
    text = re.sub(r"^.*?/generate_code/", "", text) if "/generate_code/" in text else re.sub(r"^generate_code/", "", text)
    text = re.sub(r"/{2,}", "/", text)
    if text.endswith("/"):
        raise FidelityError(f"SOURCE_FILE_IS_DIRECTORY: {value!r}")
    if not text or text.startswith("/") or any(p in {"", ".", ".."} for p in text.split("/")):
        raise FidelityError(f"SOURCE_FILE_INVALID: {value!r}; use an exact path relative to generate_code")
    return PurePosixPath(text).as_posix()


_QUOTED = re.compile(r'[\"\u201c]([^\"\u201d]+)[\"\u201d]')
_PAREN = re.compile(r"\s*\([^)]*\)?\s*$")
_FLOATING = re.compile(r"^(?:Algorithm|Alg\.?|Table|Tab\.?|Figure|Fig\.?|Equation|Eq\.?|Theorem|Lemma)\s*\(?\s*([A-Za-z0-9.]+)\)?$", re.I)


def split_pointers(refs: str) -> list[str]:
    """One ``Source:`` line's references, as the planner writes them: ``§4.1 (Practical Implementation), Appendix A
    Table 3, Addendum "Additional Details on GC-BC / GC-IQL / OPAL / SF and FB Baselines"`` — quoted titles are one
    reference each (whatever they contain), the rest splits on commas / semicolons / slashes / "and"."""
    out: list[str] = []
    rest = refs.strip().rstrip(".")
    for q in _QUOTED.findall(rest):
        out.append(q.strip())
    rest = _QUOTED.sub(" ", rest)
    # split only outside parentheses: "§3.2 (inner-loop step and acceleration trick), §5.2 (Adam lr=0.001 for
    # F-MNIST/SVHN)" is two references, not six (lbcs)
    pieces, depth, cur = [], 0, ""
    i = 0
    while i < len(rest):
        ch = rest[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if depth == 0 and (ch in ",;/&" or rest[i:i + 5] == " and "):
            pieces.append(cur)
            cur = ""
            i += 5 if rest[i:i + 5] == " and " else 1
            continue
        cur += ch
        i += 1
    pieces.append(cur)
    for ref in pieces:
        ref = ref.strip().strip("`[]").strip()
        if ref.startswith("(") and ref.endswith(")"):
            ref = ref[1:-1].strip()
        if ref:
            out.append(ref)
    return out


_NUMBER_HEAD = re.compile(r"^(?:§|section\s+|sec\.\s*|appendix\s+)?\s*(?P<num>(?:[0-9]+(?:\.[0-9]+)*|[A-Z](?:\.[0-9]+)*))\b", re.I)
_RANGE = re.compile(r"^§?\s*(?P<a>[A-Z]?[0-9]+(?:\.[0-9]+)*)\s*[-–—]\s*§?\s*(?P<b>[A-Z]?[0-9]+(?:\.[0-9]+)*)$")
_FLOAT_ANY = re.compile(r"\b(?P<kind>Algorithm|Alg\.?|Table|Tab\.?|Figure|Fig\.?|Equation|Eq\.?|Theorem|Lemma|Definition)\s*\(?\s*(?P<num>[A-Za-z0-9.]+)\)?", re.I)


def resolve_pointers(index: PaperIndex, ref: str) -> list[str]:
    """Every section a pointer names (a range names several), in the spellings planners use: the whole text; the
    leading section number alone (``§3.4 Initialization``, ``§2 Objective formulations``); without a parenthetical
    (``§4.1 (Practical Implementation)``); a range (``§6.1-§6.4``); a floating ``Algorithm 1`` / ``Eq. (14)`` /
    ``Table 3`` → the first section whose body carries it; a quoted title → the heading of that name."""
    text = ref.strip().strip("`[]").strip()
    m = _RANGE.match(text)
    if m:
        a, b = m.group("a"), m.group("b")
        prefix = a.rsplit(".", 1)[0] if "." in a else ""
        found = []
        for sec in index.sections:
            n = sec.number
            if not n:
                continue
            if n in (a, b) or (prefix and n.startswith(prefix + ".") and a <= n <= b):
                found.append(sec.section_id)
        if found:
            return found
    candidates = [text]
    stripped = _PAREN.sub("", text).strip()
    if stripped and stripped != text:
        candidates.append(stripped)
    head = _NUMBER_HEAD.match(text)
    if head:
        candidates.append(head.group("num"))
    inner = re.search(r"\(([^)]+)\)?", text)
    if inner:
        candidates.append(inner.group(1).strip())
    for cand in candidates:
        try:
            return [exact_section(index, cand).section_id]
        except FidelityError:
            continue
    m = _FLOAT_ANY.search(text)
    if m:
        kind = m.group("kind").rstrip(".").lower()[:3]
        needle = re.compile(re.escape(kind) + r"\w*\.?\s*\(?\s*" + re.escape(m.group("num")) + r"\)?(?![0-9.])", re.I)
        homes = [sec for sec in index.sections if sec.number and needle.search(sec.body)]
        if homes:
            return [homes[0].section_id]  # the first section that carries "Algorithm 1" / "Eq. (14)" is where it is stated
    return []


def resolve_pointer(index: PaperIndex, ref: str) -> str | None:
    found = resolve_pointers(index, ref)
    return found[0] if found else None


def _section_key(text: str) -> str:
    """Pointer spelling folded: prefix words, case, whitespace, the dot the paper prints after a number."""
    text = re.sub(r"^(?:§|section\s+|sec\.\s*|appendix\s+)", "", text.strip(), flags=re.I)
    text = re.sub(r"^([0-9]+(?:\.[0-9]+)*|[A-Z](?:\.[0-9]+)*)\.(?=\s|$)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip().rstrip(".").lower()


def exact_section(index: PaperIndex, pointer: str):
    wanted = _section_key(pointer)
    matches = []
    for s in index.sections:
        keys = {s.number, s.title, s.label, s.section_id}
        if s.number and s.title:
            keys.add(f"{s.number} {s.title}")
        if wanted and wanted in {_section_key(k) for k in keys if k}:
            matches.append(s)
    if len(matches) > 1:  # bbox: a "#### 1" inside a worked example folds to "1" like "1 Introduction" — the numbered heading wins
        numbered = [s for s in matches if s.number]
        matches = numbered if len(numbered) == 1 else matches
    if len(matches) != 1:
        raise FidelityError(f"SOURCE_SECTION_NOT_UNIQUE: {pointer!r}")
    return matches[0]


# ---------------------------------------------------------------------------------------------
# the blueprint, read back
# ---------------------------------------------------------------------------------------------


def _plan_root(plan_text: str) -> dict:
    try:
        value = yaml.safe_load(extract_yaml_candidate(plan_text))
    except yaml.YAMLError as exc:
        raise FidelityError(f"SOURCE_PLAN_YAML_INVALID: {exc}") from exc
    root = value.get("complete_reproduction_plan", value) if isinstance(value, dict) else None
    if not isinstance(root, dict):
        raise FidelityError("SOURCE_PLAN_INVALID: no complete_reproduction_plan mapping")
    return root


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n\n".join(_as_text(v) for v in value)
    if isinstance(value, dict):
        return "\n".join(f"{k}: {_as_text(v)}" for k, v in value.items())
    return "" if value is None else str(value)


def planned_files(plan_text: str) -> list[str]:
    """Every file the blueprint's ``file_structure`` lists, as paths relative to generate_code. A tree drawing carries
    nesting as indentation; a plain list carries full paths; both are read. Directories are not files."""
    root = _plan_root(plan_text)
    text = _as_text(root.get("file_structure"))
    files: list[str] = []
    stack: list[tuple[int, str]] = []  # (indent, directory) for tree drawings
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        cleaned = re.sub(r"[│├└─|]", " ", line)
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        token = cleaned.strip().split("#", 1)[0].strip().strip("`*- ")
        if not token:
            continue
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if token.endswith("/"):
            stack.append((indent, token))
            continue
        m = _FILE_LINE.match(re.sub(r"^(?:\./)+", "", token))
        if not m:
            continue
        path = m.group("path")
        ext = path.rsplit(".", 1)[-1].lower()
        if ext not in _CODE_EXT and "/" not in path:
            continue
        if "/" not in path and stack:
            path = "".join(d for _, d in stack) + path
        try:
            path = relative_file(path)
        except FidelityError:
            continue
        if path not in files:
            files.append(path)
    return files


def _paragraphs(text: str) -> list[str]:
    """Section 2 split into component paragraphs: on blank lines and on numbered / bulleted heads."""
    parts: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        head = re.match(r"^\s*(?:\d+[.)]|[-*•]|###?)\s+\S", line) and not line.startswith("     ")
        if (not line.strip() or head) and current:
            parts.append("\n".join(current))
            current = []
        if line.strip():
            current.append(line)
    if current:
        parts.append("\n".join(current))
    return parts


def _files_in(text: str, files: list[str]) -> list[str]:
    """Planned files a paragraph names — by full path, by a unique basename, or by a directory (``models/`` binds every
    planned file under a ``models`` directory: lbcs's Section 2 names five components by directory)."""
    by_name: dict[str, list[str]] = {}
    for f in files:
        by_name.setdefault(f.rsplit("/", 1)[-1], []).append(f)
    found: list[str] = []
    for d in _DIR_TOKEN.findall(text):
        d = d.strip("/")
        for f in files:
            if (f.startswith(d + "/") or ("/" + d + "/") in f) and f not in found:
                found.append(f)
    for token in _PATH_TOKEN.findall(text):
        token = token.strip(".")
        try:
            candidate = relative_file(token)
        except FidelityError:
            continue
        hit = None
        if candidate in files:
            hit = candidate
        else:
            for f in files:
                if f.endswith("/" + candidate):
                    hit = f
                    break
            if hit is None and "/" not in candidate and len(by_name.get(candidate, [])) == 1:
                hit = by_name[candidate][0]
        if hit and hit not in found:
            found.append(hit)
    return found


def compile_manifest(plan_text: str, paper: bytes) -> dict:
    """The blueprint read back: ``files[path] → sections`` from Section 2's paragraphs, ``glue`` for the rest,
    ``unmatched`` for pointers that name no heading, ``pointers`` = how many ``Source:`` lines the plan has at all."""
    root = _plan_root(plan_text)
    index = PaperIndex.from_markdown(paper.decode("utf-8"))
    files = planned_files(plan_text)
    bound: dict[str, list[str]] = {}
    unmatched: list[dict[str, str]] = []
    pointers = 0
    orphan_pointers = 0
    for para in _paragraphs(_as_text(root.get("implementation_components"))):
        sections: list[str] = []
        for m in _SOURCE_LINE.finditer(para):
            pointers += 1
            for ref in split_pointers(m.group("refs")):
                if _NOT_SPECIFIED.search(ref):
                    continue
                found = resolve_pointers(index, ref)
                if not found:
                    unmatched.append({"pointer": ref, "paragraph": para[:80]})
                    continue
                for sid in found:
                    if sid not in sections:
                        sections.append(sid)
        if not sections:
            continue
        named = _files_in(para, files)
        if not named:
            orphan_pointers += 1
            continue
        for path in named:
            for sid in sections:
                if sid not in bound.setdefault(path, []):
                    bound[path].append(sid)
    manifest = {
        "schema_version": VERSION, "paper_sha256": digest(paper), "plan_sha256": digest(plan_text.encode()),
        "files": {p: {"sections": bound.get(p, [])} for p in files},
        "glue": [p for p in files if p not in bound],
        "unmatched": unmatched, "pointers": pointers, "orphan_pointers": orphan_pointers,
        "sections": {s.section_id: {"heading": s.label, "sha256": digest(s.body.encode())} for s in index.sections},
    }
    return manifest


def freeze_manifest(task_dir: Path, plan_text: str) -> dict:
    document = compile_manifest(plan_text, (task_dir / "paper.md").read_bytes())
    path = task_dir / MANIFEST
    if path.is_file() and (task_dir / TRACE).is_file() and json.loads(path.read_text()) != document:
        raise FidelityError("SOURCE_PLAN_CHANGED_AFTER_IMPLEMENTATION: start a new run")
    _save(path, document)
    return document


def _save(path: Path, document: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded(document))
    temporary.replace(path)


def load_manifest(task_dir: Path) -> dict:
    try:
        saved = json.loads((task_dir / MANIFEST).read_text())
        expected = compile_manifest((task_dir / "initial_plan.txt").read_text(), (task_dir / "paper.md").read_bytes())
    except (OSError, ValueError) as exc:
        raise FidelityError(f"SOURCE_MANIFEST_REQUIRED_OR_INVALID: {exc}") from exc
    if saved != expected:
        raise FidelityError("SOURCE_MANIFEST_CHANGED")
    return saved


# ---------------------------------------------------------------------------------------------
# the session: receipts, read-before-write
# ---------------------------------------------------------------------------------------------


class FidelitySession:
    """A fresh model context. Reads from an earlier process never authorise a new write."""

    def __init__(self, task_dir: Path, code_dir: Path):
        self.task_dir, self.code_dir = Path(task_dir), Path(code_dir).resolve()
        self.manifest = load_manifest(self.task_dir)
        self.manifest_sha = digest(encoded(self.manifest))
        self.index = PaperIndex.from_file(self.task_dir / "paper.md")
        self.session = uuid.uuid4().hex
        self.turn, self.epoch = 0, 0
        self.pending: list[dict] = []
        self.events = _read_trace(self.task_dir, self.manifest_sha)

    def next_turn(self) -> None:
        self.turn += 1

    def clear_context(self) -> None:
        self.epoch += 1
        self.pending.clear()

    def path(self, value: str) -> str:
        candidate = Path(value)
        target = candidate if candidate.is_absolute() else self.code_dir / relative_file(value)
        try:
            path = target.resolve().relative_to(self.code_dir).as_posix()
        except ValueError as exc:
            raise FidelityError("SOURCE_FILE_OUTSIDE_CODE_ROOT") from exc
        if target.is_symlink() or target.absolute() != target.resolve():
            raise FidelityError("SOURCE_FILE_ALIAS_REFUSED")
        return path

    def sections_for(self, file_path: str) -> list[str]:
        """The sections a planned file must read; ``[]`` for glue and for files the plan did not list (those are
        the coding agent's own additions — allowed, unbound)."""
        entry = self.manifest["files"].get(self.path(file_path))
        return list(entry["sections"]) if entry else []

    def pages(self, section_id: str) -> int:
        body = exact_section(self.index, section_id).body
        return max(1, -(-len(body) // DEFAULT_PART_CHARS))

    def unread_pages(self, file_path: str, *, earlier_turn_only: bool = False) -> list[tuple[str, int]]:
        """Every (section, page) a file is bound to and has not read this context — the whole of every bound section
        (owner 09-21: a pointer names a section, so the section is read, not a page of it; fre-t17 read page 1 of
        three-page §4.1 and wrote). ``earlier_turn_only`` counts only reads from before the current model turn."""
        path = self.path(file_path)
        have = {(r["section_id"], r["part"]) for r in self.pending
                if r["file_path"] == path and (not earlier_turn_only or r["turn"] < self.turn)}
        return [(sid, page) for sid in self.sections_for(path) for page in range(1, self.pages(sid) + 1) if (sid, page) not in have]

    def next_unread(self, file_path: str) -> tuple[str, int] | None:
        left = self.unread_pages(file_path)
        return left[0] if left else None

    def _event(self, payload: dict) -> dict:
        event = {**payload, "seq": len(self.events), "session": self.session, "turn": self.turn,
                 "epoch": self.epoch, "previous": self.events[-1]["sha256"] if self.events else self.manifest_sha}
        event["sha256"] = digest(encoded(event))
        self.events.append(event)
        _save(self.task_dir / TRACE, {"schema_version": VERSION, "manifest_sha256": self.manifest_sha, "events": self.events})
        return event

    def read(self, *, file_path: str = "", section: str = "", part: int = 1) -> str:
        """One page of one section, as a receipt for ``file_path`` (or a free read when no file is given)."""
        path = self.path(file_path) if file_path else ""
        bound = self.sections_for(path) if path else []
        wanted = exact_section(self.index, section) if section else None
        if wanted is None:
            if not path:
                raise FidelityError("SOURCE_SECTION_REQUIRED")
            nxt = self.next_unread(path)
            if nxt is None and not bound:
                raise FidelityError(f"SOURCE_GLUE_FILE: {path} has no Source pointer in the plan; write it without a read")
            sid, part = nxt if nxt is not None else (bound[0], 1)
            wanted = exact_section(self.index, sid)
        parts = max(1, -(-len(wanted.body) // DEFAULT_PART_CHARS))
        if type(part) is not int or not 1 <= part <= parts:
            raise FidelityError(f"SOURCE_PART_OUT_OF_RANGE: {wanted.section_id} has {parts} page(s)")
        start = (part - 1) * DEFAULT_PART_CHARS
        end = min(len(wanted.body), start + DEFAULT_PART_CHARS)
        result = {"section_id": wanted.section_id, "heading": wanted.label, "file_path": path, "bound": wanted.section_id in bound,
                  "paper_sha256": self.manifest["paper_sha256"], "section_sha256": digest(wanted.body.encode()),
                  "start": start, "end": end, "part": part, "parts": parts, "content": wanted.body[start:end]}
        event = self._event({"kind": "read", **result, "result_sha256": digest(encoded(dict(result)))})
        if path:
            self.pending.append(event)
        head = {k: v for k, v in result.items() if k != "content"}
        if path:
            left = self.unread_pages(path)
            head["still_unread_for_file"] = [f"§{s} p{p}" for s, p in left[:12]] or "none — every bound page of this file is read; write it in your next reply"
        return json.dumps(head, ensure_ascii=False) + "\n\n" + result["content"]

    def authorize(self, file_path: str) -> list[int]:
        """Receipt numbers that authorise writing ``file_path`` now; raises with the sections still unread."""
        path = self.path(file_path)
        reads = [r for r in self.pending if r["file_path"] == path and r["turn"] < self.turn]
        missing = self.unread_pages(path, earlier_turn_only=True)
        if missing:
            raise FidelityError("SOURCE_READ_REQUIRED: " + json.dumps({"file_path": path, "pages": [f"§{s} p{p}" for s, p in missing[:20]]}, ensure_ascii=False)
                                + "; call read_paper(file_path=...) until it reports none unread, then write in a later model turn")
        return [r["seq"] for r in reads]

    def written(self, file_path: str, receipts: list[int]) -> None:
        path = self.path(file_path)
        self._event({"kind": "write", "file_path": path, "file_sha256": digest((self.code_dir / path).read_bytes()),
                     "reads": receipts, "planned": path in self.manifest["files"]})
        self.pending = [r for r in self.pending if r["file_path"] != path]


def resolve_read_args(session: FidelitySession, args: dict) -> str | None:
    """Fit a ``read_paper`` call onto the file's obligations, rewriting ``args`` in place; the refusal text otherwise."""
    file_path = str(args.get("file_path") or "")
    if not file_path:
        return None
    try:
        path = session.path(file_path)
    except FidelityError as exc:
        return str(exc)
    bound = session.sections_for(path)
    if not bound:
        return f"SOURCE_GLUE_FILE: {path} has no Source pointer in the plan; write it without a read"
    section = str(args.get("section") or "")
    if section:
        try:
            sid = exact_section(session.index, section).section_id
        except FidelityError:
            sid = ""
        section = sid if sid in bound else ""
    if section and args.get("part"):
        args["section"], args["file_path"] = section, path
        return None
    nxt = session.next_unread(path)
    if nxt is None:
        nxt = (section or bound[0], 1)
    elif section and section != nxt[0]:
        pages_left = [p for s, p in session.unread_pages(path) if s == section]
        nxt = (section, pages_left[0]) if pages_left else nxt
    args["section"], args["part"], args["file_path"] = nxt[0], nxt[1], path
    return None


def _read_trace(task_dir: Path, manifest_sha: str) -> list[dict]:
    path = task_dir / TRACE
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
        if data["schema_version"] != VERSION or data["manifest_sha256"] != manifest_sha:
            raise FidelityError("SOURCE_TRACE_SCOPE_CHANGED")
        events = data["events"]
        previous = manifest_sha
        for i, event in enumerate(events):
            if (event["seq"] != i or event["previous"] != previous
                    or event["sha256"] != digest(encoded({k: v for k, v in event.items() if k != "sha256"}))):
                raise FidelityError("SOURCE_TRACE_CHANGED")
            previous = event["sha256"]
        return events
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FidelityError(f"SOURCE_TRACE_INVALID: {exc}") from exc


# ---------------------------------------------------------------------------------------------
# the audit: a record, not a gate
# ---------------------------------------------------------------------------------------------


def audit(task_dir: Path, code_dir: Path) -> dict:
    """Replay the trace against the manifest and the final bytes; every finding is a row in ``violations``, and
    ``passed`` is just ``not violations``. Nothing here fails a phase."""
    manifest = load_manifest(task_dir)
    events = _read_trace(task_dir, digest(encoded(manifest)))
    violations: list[str] = []
    reads: dict[int, dict] = {}
    latest: dict[str, dict] = {}
    sections = manifest["sections"]
    for event in events:
        if event["kind"] == "read":
            info = sections.get(event["section_id"])
            if info is None or info["sha256"] != event["section_sha256"] or event["paper_sha256"] != manifest["paper_sha256"]:
                violations.append(f"read #{event['seq']}: section bytes differ from the manifest")
            reads[event["seq"]] = event
        elif event["kind"] == "write":
            path = event["file_path"]
            selected = [reads[i] for i in event["reads"] if i in reads]
            if len(selected) != len(event["reads"]) or any(
                r["file_path"] != path or r["session"] != event["session"] or r["epoch"] != event["epoch"] or r["turn"] >= event["turn"]
                for r in selected
            ):
                violations.append(f"write of {path}: cited receipts are not this file's earlier-turn reads")
            have = {(r["section_id"], r["part"]) for r in selected}
            parts_of = {r["section_id"]: r["parts"] for r in reads.values()}
            for sid in (manifest["files"].get(path) or {"sections": []})["sections"]:
                total = parts_of.get(sid, 1)
                unread = [p for p in range(1, total + 1) if (sid, p) not in have]
                if unread:
                    violations.append(f"write of {path}: bound section {sid} not fully read before it (pages {unread})")
            latest[path] = event
    actual = {p.relative_to(code_dir).as_posix(): p for p in Path(code_dir).rglob("*") if p.is_file()
              and not set(p.relative_to(code_dir).parts) & {".git", "__pycache__", ".pytest_cache"}}
    # the plan's tree drawing and the coder's tree differ in spelling more than in substance (fre-t17: README.md and
    # requirements.txt drawn at the top, written under fre/; eight package __init__.py the structure agent adds):
    # a planned file present under another directory is not missing, and package markers are never unplanned
    actual_names = {p.rsplit("/", 1)[-1] for p in actual}
    planned_names = {p.rsplit("/", 1)[-1] for p in manifest["files"]}
    missing = sorted(p for p in set(manifest["files"]) - set(actual) if p.rsplit("/", 1)[-1] not in actual_names)
    unplanned = sorted(p for p in set(actual) - set(manifest["files"])
                       if p.rsplit("/", 1)[-1] != "__init__.py" and p.rsplit("/", 1)[-1] not in planned_names)
    unverified = sorted(p for p, f in actual.items() if p in manifest["files"] and (p not in latest or digest(f.read_bytes()) != latest[p]["file_sha256"]))
    for label, rows in (("planned but absent", missing), ("written outside the plan", unplanned), ("planned file whose final bytes no recorded write produced", unverified)):
        for p in rows:
            violations.append(f"{label}: {p}")
    paper_files = [p for p, e in manifest["files"].items() if e["sections"]]
    return {"passed": not violations, "files": len(actual), "planned": len(manifest["files"]), "paper_files": len(paper_files),
            "glue_files": len(manifest["glue"]), "read_receipts": len(reads), "characters_read": sum(len(r["content"]) for r in reads.values()),
            "unmatched_pointers": len(manifest["unmatched"]), "missing": missing, "unplanned": unplanned, "unverified": unverified,
            "violations": violations, "paper_sha256": manifest["paper_sha256"], "manifest_sha256": digest(encoded(manifest))}


__all__ = [
    "MANIFEST", "TRACE", "VERSION", "FidelityError", "FidelitySession", "audit", "compile_manifest", "digest", "encoded",
    "exact_section", "freeze_manifest", "load_manifest", "planned_files", "relative_file", "resolve_pointer", "resolve_pointers",
    "resolve_read_args", "split_pointers",
]
