"""Paper fidelity for the blueprint and the coding loop (Paper2Code line, ADR 0004; ``DEEPCODE_PAPER_FIDELITY=1``).

Two things, one switch, default off (upstream behaviour byte-identical when the variable is unset):

* **Planning**: the planner (and the fan-out algorithm-analysis agent) are told to write, in every Section 2
  paragraph that implements something the paper specifies, the file path(s) and a ``Source: §x.y`` pointer to the
  section(s) the coding agent must read — and *not* to copy the paper's formulas or LaTeX into the plan (ADR 0003's
  verbatim quotes are gone: the plan is a summary, the paper is the specification). The host reads those pointers
  back mechanically into ``source_manifest.json`` (``source_fidelity.compile_manifest``): file → bound sections.
* **Implementation**: the coding agent gets ``read_paper`` — a deterministic look-up over the paper's own headings
  (``\\section*{4.1. …}``, ``## …``) that returns a section's text with its display equations, a few thousand
  characters a page. ``read_paper(file_path=…)`` walks that file's unread (section, page) list; ``write_file`` is
  refused until every page of every bound section was read in an earlier model turn. The memory agent's clean
  slate after every ``write_file`` is untouched, so a read-back lives exactly as long as the file it served.

The index is built from the same ``paper.md`` the planner read; nothing is retrieved by relevance scoring.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from apps.v2.agent_engine.paper2code.seams.agent_runtime import Tool, ToolResult, tool_parameters

FIDELITY_ENV = "DEEPCODE_PAPER_FIDELITY"
#: text the line appends to the planner prompt on a re-plan (phase_plan: obligations no file claimed); unset = none
PLANNER_FEEDBACK_ENV = "DEEPCODE_PLANNER_FEEDBACK"
READ_PAPER_TOOL = "read_paper"
DEFAULT_PART_CHARS = 3000
MAX_PART_CHARS = 8000


def fidelity_enabled() -> bool:
    return os.environ.get(FIDELITY_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------------------------
# the paper index
# ---------------------------------------------------------------------------------------------

_LATEX_HEADING = re.compile(r"^\\(sub)*section\*?\{(?P<title>[^}]*)\}\s*$")
_MD_HEADING = re.compile(r"^(?P<hashes>#{1,4})\s+(?P<title>.+?)\s*#*\s*$")
_NUMBER_PREFIX = re.compile(r"^\s*(?P<num>(?:[A-Z]|\d+)(?:\.\d+)*)(?:\.\s*|\s+)(?P<rest>\S.*)$")  # "4.2 Title", "4.2. Title", "2.1.Title" (adaptive-pruning)
_DISPLAY_EQ = re.compile(r"\\\[(.*?)\\\]|\$\$(.*?)\$\$", re.S)
_FULLWIDTH = str.maketrans({"．": ".", "　": " "})


@dataclass(slots=True)
class Section:
    number: str  # "4.1", "A", "" when the heading carries no number
    title: str
    body: str
    level: int
    equations: list[str] = field(default_factory=list)

    @property
    def section_id(self) -> str:
        return self.number or self.title

    @property
    def label(self) -> str:
        return f"§{self.number} {self.title}".strip() if self.number else self.title


@dataclass(slots=True)
class PaperIndex:
    sections: list[Section]

    @classmethod
    def from_markdown(cls, text: str) -> "PaperIndex":
        lines = text.splitlines()
        heads: list[tuple[int, int, str]] = []  # (line index, level, raw title)
        for i, line in enumerate(lines):
            m = _LATEX_HEADING.match(line.strip())
            if m:
                heads.append((i, 1 + (len(m.group(1) or "") // 3), m.group("title")))
                continue
            m = _MD_HEADING.match(line)
            if m and not line.startswith("#!"):
                heads.append((i, len(m.group("hashes")), m.group("title")))
        sections: list[Section] = []
        if heads and heads[0][0] > 0:
            front = "\n".join(lines[: heads[0][0]]).strip()
            if front:
                sections.append(Section(number="", title="(front matter)", body=front, level=0, equations=_equations(front)))
        for k, (start, level, raw) in enumerate(heads):
            end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
            body = "\n".join(lines[start + 1 : end]).strip()
            title = raw.translate(_FULLWIDTH).strip()
            m = _NUMBER_PREFIX.match(title)
            number, clean = (m.group("num"), m.group("rest").strip()) if m else ("", title)
            sections.append(Section(number=number, title=clean, body=body, level=level, equations=_equations(body)))
        if not sections and text.strip():
            sections.append(Section(number="", title="(paper)", body=text.strip(), level=0, equations=_equations(text)))
        return cls(sections)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "PaperIndex":
        return cls.from_markdown(Path(path).read_text(encoding="utf-8", errors="replace"))

    # -- look-ups ----------------------------------------------------------------------------

    def outline(self) -> str:
        rows = []
        for s in self.sections:
            eq = f"  [{len(s.equations)} display equation{'s' if len(s.equations) != 1 else ''}]" if s.equations else ""
            rows.append(f"{'  ' * max(0, s.level - 1)}{s.label}{eq}")
        return "\n".join(rows)

    def find(self, section: str | None = None, query: str | None = None) -> list[Section]:
        """Sections matching a pointer (``"4.1"``, ``"§4.1"``, ``"Section 4.1"``, ``"4.1. Aggregating data"``, a
        title fragment) or, failing that, ranked by keyword hits for ``query``."""
        wanted = (section or "").translate(_FULLWIDTH).strip()
        wanted = re.sub(r"^(§|section|sec\.?|appendix)\s*", "", wanted, flags=re.I).strip().rstrip(".")
        if wanted:
            num = re.match(r"^([A-Z]|\d+)(\.\d+)*$", wanted)
            if num:
                exact = [s for s in self.sections if s.number == wanted]
                if exact:
                    return exact
                return [s for s in self.sections if s.number.startswith(wanted + ".")]
            head = re.match(r"^((?:[A-Z]|\d+)(?:\.\d+)*)\.?\s+(.+)$", wanted)
            if head and any(s.number == head.group(1) for s in self.sections):
                return [s for s in self.sections if s.number == head.group(1)]
            low = wanted.lower()
            hits = [s for s in self.sections if low in s.title.lower()]
            if hits:
                return hits
            query = query or wanted
        if query:
            words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query.lower())
            if not words:
                return []
            scored = []
            for s in self.sections:
                title, body = s.title.lower(), s.body.lower()
                score = sum(3 for w in words if w in title) + sum(min(body.count(w), 5) for w in words)
                if score:
                    scored.append((score, s))
            scored.sort(key=lambda t: -t[0])
            return [s for _, s in scored[:3]]
        return []

    def render(self, section: Section, part: int = 1, part_chars: int = DEFAULT_PART_CHARS) -> str:
        body = section.body or "(empty section)"
        parts = max(1, -(-len(body) // part_chars))
        part = min(max(1, part), parts)
        chunk = body[(part - 1) * part_chars : part * part_chars]
        head = f"{section.label} — part {part}/{parts}" if parts > 1 else section.label
        eqs = ""
        if section.equations and part == 1:
            listed = "\n".join(f"  ({i}) {e[:400]}" for i, e in enumerate(section.equations, 1))
            eqs = f"\n\nDisplay equations in this section (verbatim, numbered here by order):\n{listed}"
        more = f"\n\n[call read_paper(section=\"{section.number or section.title}\", part={part + 1}) for the rest]" if part < parts else ""
        return f"{head}\n\n{chunk}{eqs}{more}"


def _equations(body: str) -> list[str]:
    out = []
    for m in _DISPLAY_EQ.finditer(body):
        eq = " ".join((m.group(1) or m.group(2) or "").split())
        if eq:
            out.append(eq)
    return out


# ---------------------------------------------------------------------------------------------
# the tool
# ---------------------------------------------------------------------------------------------

@tool_parameters(
    {
        "type": "object",
        "properties": {
            "section": {
                "type": "string",
                "description": 'Section pointer from the plan\'s "Source" notes: "4.1", "§4.1", "Section 4.2", "B.3", or a heading fragment such as "Symmetric aggregation". Empty string = the outline of the paper.',
            },
            "query": {"type": "string", "description": "Keywords to locate the section when the pointer is unknown (e.g. \"off-policy critic loss\")."},
            "part": {"type": "integer", "description": "Page of a long section (1-based); the first page says how many there are.", "minimum": 1},
            "file_path": {"type": "string", "description": "The planned file this read is for: returns the next section that file is bound to (its plan paragraph's `Source:` line) and records the receipt the file's write needs."},
        },
        "required": [],
    }
)
class ReadPaperTool(Tool):
    """Verbatim read-back of one section of the paper (headings + display equations), a few thousand characters at a time."""

    def __init__(self, index: PaperIndex, *, on_call: Callable[[dict[str, Any], str], str | None] | None = None,
                 validate_call: Callable[[dict[str, Any]], str | None] | None = None,
                 part_chars: int = DEFAULT_PART_CHARS):
        self._index = index
        self._on_call = on_call
        self._validate_call = validate_call
        self._part_chars = max(500, min(int(part_chars), MAX_PART_CHARS))
        self.calls = 0

    @property
    def name(self) -> str:
        return READ_PAPER_TOOL

    @property
    def description(self) -> str:
        return (
            "Read the paper's ORIGINAL text, one page (3000 chars) at a time. `file_path=<planned file>` returns the "
            "next unread page of the sections that file is bound to by its `Source: §…` line in the plan, and says what "
            "is still unread — call again until none (the write of that file is refused until every page of every "
            "bound section was read). `section=<number or heading>` (+ `part`) reads any section freely. Call with no "
            "arguments for the outline."
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, section: str = "", query: str = "", part: int = 1,
                      file_path: str = "",
                      validate_call: Callable[[dict[str, Any]], str | None] | None = None,
                      **_: Any) -> str:
        self.calls += 1
        section, query = (section or "").strip(), (query or "").strip()
        args = {"section": section, "query": query, "part": part, "file_path": file_path}
        validator = validate_call or self._validate_call
        if validator is not None and (section or query or file_path):
            error = validator(args)
            if error:
                return ToolResult(error, is_error=True, metadata={"paper_readback": "source_binding"})
            section, part = str(args.get("section") or ""), int(args.get("part") or 1)  # the validator resolves the file's next unread (section, page)
        if not section and not query:
            text = "Paper outline (call read_paper(section=<number or heading>) for a section):\n" + self._index.outline()
        else:
            hits = self._index.find(section=section or None, query=query or None)
            if not hits:
                text = f"No section matches {section or query!r}. Outline:\n" + self._index.outline()
            elif len(hits) == 1 or section:
                text = self._index.render(hits[0], part=int(part or 1), part_chars=self._part_chars)
                if len(hits) > 1:
                    text += "\n\nOther matches: " + "; ".join(h.label for h in hits[1:4])
            else:
                text = "Closest sections for that query (call again with section=<number>):\n" + "\n".join(f"- {h.label}" for h in hits)
        if self._on_call:
            try:
                error = self._on_call(args, text)
            except Exception as exc:
                error = f"SOURCE_READ_REJECTED: {type(exc).__name__}: {exc}"
            if error:
                return ToolResult(error, is_error=True, metadata={"paper_readback": "source_binding"})
        return text


def load_paper_index(paper_dir: str | os.PathLike[str]) -> PaperIndex | None:
    """The task directory's ``paper.md`` (what the planner read) as an index; None when there is none."""
    path = Path(paper_dir) / "paper.md"
    if not path.is_file():
        cands = sorted(p for p in Path(paper_dir).glob("*.md") if not p.name.endswith("implement_code_summary.md"))
        if not cands:
            return None
        path = cands[0]
    try:
        return PaperIndex.from_file(path)
    except OSError:
        return None


# ---------------------------------------------------------------------------------------------
# prompt addenda (appended only when the switch is on)
# ---------------------------------------------------------------------------------------------

PLANNING_FIDELITY_ADDENDUM = """

# SOURCE POINTERS (MANDATORY — read before writing Section 2)
The coding agent that implements this plan does NOT see the paper. For every file it writes, it reads the paper
sections you point at here, and nothing else. So in Section 2 (implementation_components), for EVERY component that
implements something the paper specifies — a formula, loss, update rule, sampling distribution, architecture detail,
hyper-parameter group, dataset / evaluation protocol — write, in that component's paragraph:
- the exact file path(s) from file_structure that implement it, and
- a line `Source: §<section number or heading>[, §<another>]` naming the paper section(s) the coding agent must read
  before writing those files (the section that states the formula / the setting; add the appendix section when the
  details live there).
Example of a paragraph that is correct:
    3) FRE objective (fre/losses/fre_objective.py)
       Variational lower bound of the information bottleneck: decoder log-likelihood minus beta * KL to u(z).
       Source: §4.1, §B
Glue files (package markers, config loaders, plotting, README) need no Source line. Do not copy the paper's LaTeX or
text into the plan — describe what the component computes and point at the section; the coding agent reads the
original. Say `Source: not specified in the paper` for a value the paper leaves open, and choose a default there.
A plan whose Section 2 has no Source lines is rejected and planned again.
"""

ANALYSIS_FIDELITY_ADDENDUM = """

# SOURCE POINTERS (MANDATORY)
For every formula, update rule, loss, sampling distribution and hyper-parameter group you describe, name the paper
section it comes from as `Source: §<section number or heading>`; do not paraphrase an equation as if it were exact,
and do not fill in values the paper does not state.
"""

IMPLEMENT_SYSTEM_ADDENDUM = """

**PAPER READ-BACK (MANDATORY for files the plan points at)**:
- The plan marks the files that implement something the paper specifies with `Source: §…`. Before `write_file` for
  such a file, call `read_paper(file_path="<exact planned path>")` — each call returns the next unread page of the
  sections that file is bound to and lists what is still unread; keep calling until it reports none. The WHOLE of
  every bound section is read, not one page of it. Implement from the ORIGINAL text: the plan is a summary, the paper
  is the specification.
- A write in the same model response as the read is rejected (SOURCE_READ_REQUIRED lists the sections still
  unread); read first, wait for the result, then write in the next response.
- Follow the paper's equation exactly (norms, exponents, expectations, ratios, index ranges, constants); where the
  paper and the plan disagree, the paper wins. Where the paper is silent, choose a sensible default and note it.
- Files without a Source line in the plan (glue) need no read; `read_paper` tells you so.
- Development cycle for a pointed file: `read_paper(file_path=…)` × bound sections → optional
  `search_code_references` → `write_file`.
"""

ROUND_FIDELITY_ADDENDUM = """

**Paper read-back:** if the next file's paragraph in the plan has a `Source:` line, call `read_paper(file_path="<path>")` FIRST, repeatedly until it reports nothing unread (every page of every bound section), and write the code from the paper's own text (the plan is a summary). Wait for the read results before calling `write_file`."""

GUIDANCE_FIDELITY_LINE = "   - **For a file whose plan paragraph has `Source: §…`: call `read_paper(file_path=\"<path>\")` first, again until it reports nothing unread** (every page of every bound section — the paper's own text), then implement"


# ---------------------------------------------------------------------------------------------
# what the line records (counts only; no scoring)
# ---------------------------------------------------------------------------------------------

_SOURCE_REF = re.compile(r"Source:\s*§?\s*([A-Z]|\d+)(\.\d+)*", re.I)
_INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)[^$\n]{3,}?\$(?!\$)")


def plan_fidelity_stats(plan_text: str, paper_text: str | None = None) -> dict[str, Any]:
    """How much of the paper's mathematics the blueprint carries: display / inline equations in the plan (and in the
    paper when given), ``Source: §`` pointers and how many of them name a real section. Recorded, never judged."""
    plan_display = len(_DISPLAY_EQ.findall(plan_text or ""))
    plan_inline = len(_INLINE_MATH.findall(plan_text or ""))
    refs = [m.group(0) for m in _SOURCE_REF.finditer(plan_text or "")]
    out: dict[str, Any] = {"plan_display_equations": plan_display, "plan_inline_math": plan_inline, "source_refs": len(refs)}
    if paper_text:
        index = PaperIndex.from_markdown(paper_text)
        numbers = {s.number for s in index.sections if s.number}
        resolved = 0
        for ref in refs:
            num = re.sub(r"^Source:\s*§?\s*", "", ref, flags=re.I).strip()
            if num in numbers or any(n.startswith(num + ".") for n in numbers):
                resolved += 1
        out.update({
            "paper_display_equations": sum(len(s.equations) for s in index.sections),
            "paper_inline_math": len(_INLINE_MATH.findall(paper_text)),
            "paper_sections": len(numbers),
            "source_refs_resolved": resolved,
            "structured_source_items": 0,
            "structured_source_files": 0,
        })
    return out


__all__ = [
    "ANALYSIS_FIDELITY_ADDENDUM",
    "DEFAULT_PART_CHARS",
    "FIDELITY_ENV",
    "GUIDANCE_FIDELITY_LINE",
    "IMPLEMENT_SYSTEM_ADDENDUM",
    "PLANNING_FIDELITY_ADDENDUM",
    "READ_PAPER_TOOL",
    "ROUND_FIDELITY_ADDENDUM",
    "PaperIndex",
    "ReadPaperTool",
    "Section",
    "fidelity_enabled",
    "load_paper_index",
    "plan_fidelity_stats",
]
