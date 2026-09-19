#!/usr/bin/env python3
"""Is a paper's official paper.md a usable stand-in for its PDF?  (run with DeepCode/.venv/bin/python — needs pypdf)

    DeepCode/.venv/bin/python deepcode_test/scripts/check_paper_md.py sapg pinn robust-clip

PaperBench ships paper.md as a Mathpix-style OCR of the PDF; the md-only arms (DeepCode, the DeepEvol line) never see
the PDF, the CLI arms can read it. 2026-09-19: robust-clip's official paper.md (hash-verified against upstream) has no
§2 Related Work, no §3 method (Unsupervised Adversarial Fine-Tuning for CLIP) and no §4 intro — the text jumps from the
introduction to Table 1 / §4.1, and rotated table headers are OCR garbage. The other four papers of the batch are
complete. This prints, per paper: word ratio md/pdf, the PDF's numbered top-level headings that have no counterpart in
the md, and a verdict; anything but OK means the md-only arms are handicapped on that paper.
"""
import re, sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2] / "frontier-evals" / "project" / "paperbench" / "data" / "papers"
WORD = re.compile(r"[A-Za-z]{3,}")
# a numbered heading line in the PDF text: "3. Method", "3 Method"
PDF_HEAD = re.compile(r"\n(\d)\.?\s+([A-Z][A-Za-z][A-Za-z ,\-:&]{2,60})\n")
MD_HEAD = re.compile(r"\\(?:sub)?section\*?\{([^}]*)\}|^#+\s+(.*)$", re.M)


def headings_pdf(text: str) -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    for m in PDF_HEAD.finditer(text):
        n, title = m.group(1), m.group(2).strip()
        counts[(n, title)] = counts.get((n, title), 0) + 1
    # running heads: the paper title next to a page number, i.e. the same title under three or more numbers
    by_title: dict[str, set[str]] = {}
    for (n, title) in counts:
        by_title.setdefault(title, set()).add(n)
    for (n, title), c in counts.items():
        if c <= 2 and int(n) <= 9 and len(by_title[title]) < 3 and title.lower() != "references" and (n, title) not in seen:
            seen.append((n, title))
    return seen


def main(papers: list[str]) -> int:
    bad = 0
    for p in papers:
        d = ROOT / p
        md = (d / "paper.md").read_text(encoding="utf-8", errors="replace")
        if md.startswith("version https://git-lfs"):
            print(f"{p:30s} paper.md is an LFS pointer — hydrate first (PAPERS={p} bash setup.sh)"); bad += 1; continue
        text = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(d / "paper.pdf")).pages)
        ratio = len(WORD.findall(md)) / max(1, len(WORD.findall(text)))
        md_heads = [(a or b).lower().replace("．", ".").replace("　", " ") for a, b in MD_HEAD.findall(md)]
        missing = []
        for n, title in headings_pdf(text):
            key = WORD.findall(title.lower())[:1]
            as_heading = any(h.lstrip().startswith(n) and key[0] in h for h in md_heads) if key else True
            in_body = title.lower() in md.lower()  # figure labels / list items the PDF regex mistook for headings
            if not as_heading and not in_body:
                missing.append(f"{n}. {title}")
        verdict = "OK" if ratio >= 0.9 and not missing else "SUSPECT"
        if verdict != "OK":
            bad += 1
        print(f"{p:30s} md/pdf words {ratio:.2f}  missing PDF sections: {missing or 'none'}  → {verdict}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["sapg"]))
