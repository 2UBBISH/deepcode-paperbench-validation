"""Figures put back where they stand: when the run's model can see images, every ``![](assets/…)`` of the
paper is described from the image and the description is inserted right after the reference.

PaperBench's ``paper.md`` carries formulas and tables as LaTeX text, but each figure is only an image
reference plus its caption; the engine and the provider are text-only, so the model never saw the plot —
and the experiment settings that papers only draw (sweep values, baselines compared, seeds, axis units)
were unavailable to the blueprint. This module is the intake's optional pass (``run.json.figures``:
``auto`` = only when the preflight vision probe says the vision model accepts images, ``on`` = attempt
anyway, ``off`` = never). The caliber model (DeepSeek-V4-Flash) is text-only; the pass uses a separate
vision model (``run.json.figures_model``, default ``DeepSeek-V4-Flash-Vision-Exp``) through the same
provider — the run's phase model stays pinned, this is input preparation. The probe is a PNG split into
four quadrants of random colours and asks for all four: an endpoint that silently drops the image part
and lets a text model guess passes a one-colour probe (V4-Flash answered "Red" to a red square on
2026-09-17), not this one. One model call per figure with the image and the caption, a structured description,
inserted as a marked blockquote so it is visibly *not* the paper's own text; the benchmark bytes stay in
``input/paper.raw.md`` and ``run.json.paper_sha256`` is still their hash. Descriptions are cached in
``input/figures.json`` against the raw hash, so a rerun of intake makes no new calls.

This is an input change: a run with figures described is not the same input as the baseline's
(``phases/01_intake.json → figures``); compare it only with runs of the same setting.
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIGURE_RE = re.compile(r"^!\[[^\]]*\]\((?P<path>[^)\s]+)\)\s*$")
BLOCK_START = "> **[Figure image content, transcribed by the model; not part of the paper's text]**"
CAPTION_MAX_CHARS = 600
MAX_FIGURES = 40
MAX_IMAGE_BYTES = 4 * 1024 * 1024
DESCRIPTION_MAX_TOKENS = 1500
FIGURES_CACHE_NAME = "figures.json"
RAW_INPUT_NAME = "paper.raw.md"

IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF8", "image/gif"),
    (b"RIFF", "image/webp"),
)

PROMPT = """This image is a figure from a research paper. Its caption is:

{caption}

Describe what the image itself contains, for someone who has to reproduce the paper's experiments and cannot see it. Copy, do not interpret:
- the kind of figure (line plot, bar chart, table rendered as an image, algorithm / pseudocode box, architecture diagram, schematic, screenshot);
- for a plot: each subplot's title, the x and y axis labels with units and the tick values or range, every curve or bar with its legend label, the values that can be read off (numbers, saturation points, crossings), whether there are shaded regions (seeds / confidence bands) and how many runs they suggest;
- for a table image: transcribe it as a Markdown table;
- for an algorithm or pseudocode box: transcribe it line by line;
- for an architecture diagram: the components, their labels, and the arrows between them, in order.
Write plain text or Markdown, at most 300 words. If something is unreadable, say "unreadable" rather than guessing. Do not repeat the caption."""

VISION_PROBE_PROMPT = (
    "This image is divided into four equal quadrants, each filled with one solid colour. Name the colour of each "
    "quadrant in this order: top-left, top-right, bottom-left, bottom-right. Answer with exactly four colour "
    "words separated by commas and nothing else."
)
PROBE_COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (220, 20, 20), "green": (20, 170, 40), "blue": (30, 60, 220), "yellow": (240, 220, 30),
    "black": (0, 0, 0), "white": (255, 255, 255), "orange": (245, 140, 20), "purple": (130, 30, 170),
}
PROBE_MIN_CORRECT = 3


@dataclass(frozen=True, slots=True)
class Figure:
    line_no: int  # 0-based line of the reference
    ref: str  # the path as written, e.g. assets/asset_3.jpg
    caption: str


# ---------------------------------------------------------------------------
# the paper text
# ---------------------------------------------------------------------------


def find_figures(text: str) -> list[Figure]:
    """Every ``![…](path)`` line, with the next non-empty line as its caption (truncated)."""
    lines = (text or "").splitlines()
    found: list[Figure] = []
    for i, line in enumerate(lines):
        match = FIGURE_RE.match(line.strip())
        if not match:
            continue
        caption = ""
        for later in lines[i + 1 :]:
            if later.strip():
                caption = later.strip()[:CAPTION_MAX_CHARS]
                break
        found.append(Figure(line_no=i, ref=match.group("path"), caption=caption))
    return found


def has_description(lines: list[str], line_no: int) -> bool:
    """Whether the reference at ``line_no`` is already followed by a description block."""
    for later in lines[line_no + 1 : line_no + 3]:
        if later.strip() == BLOCK_START:
            return True
        if later.strip():
            return False
    return False


def description_block(text: str) -> list[str]:
    body = [ln.rstrip() for ln in (text or "").strip().splitlines()] or ["(empty description)"]
    return [BLOCK_START, ">"] + [f"> {ln}" if ln else ">" for ln in body]


def insert_descriptions(text: str, descriptions: dict[str, str]) -> str:
    """Insert each description right after its ``![](ref)`` line; references already described are left alone."""
    lines = (text or "").splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        match = FIGURE_RE.match(line.strip())
        if not match:
            continue
        ref = match.group("path")
        if ref in descriptions and descriptions[ref] and not has_description(lines, i):
            out.append("")
            out.extend(description_block(descriptions[ref]))
    result = "\n".join(out)
    return result + ("\n" if text.endswith("\n") else "")


# ---------------------------------------------------------------------------
# images and the model
# ---------------------------------------------------------------------------


def image_mime(data: bytes) -> str | None:
    for magic, mime in IMAGE_MAGIC:
        if data.startswith(magic):
            return mime
    return None


def image_part(data: bytes, mime: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"}}


def probe_png(size: int = 64, quadrants: tuple[str, str, str, str] = ("red", "red", "red", "red")) -> bytes:
    """A small PNG with four solid-colour quadrants (top-left, top-right, bottom-left, bottom-right), stdlib only."""
    half = size // 2
    rows = []
    for y in range(size):
        left, right = (quadrants[0], quadrants[1]) if y < half else (quadrants[2], quadrants[3])
        rows.append(b"\x00" + bytes(PROBE_COLOURS[left]) * half + bytes(PROBE_COLOURS[right]) * (size - half))
    raw = b"".join(rows)

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def redact_images(messages: Any) -> Any:
    """The same messages with every ``data:`` image URL replaced by its size (for ``llm/<seq>.json``)."""
    if not isinstance(messages, list):
        return messages
    out = []
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            parts = []
            for part in message["content"]:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = str((part.get("image_url") or {}).get("url") or "")
                    if url.startswith("data:"):
                        head = url.split(",", 1)[0]
                        parts.append({"type": "image_url", "image_url": {"url": f"{head},<{len(url)} chars redacted>"}})
                        continue
                parts.append(part)
            out.append({**message, "content": parts})
        else:
            out.append(message)
    return out


def probe_quadrants(rng: random.Random | None = None) -> tuple[str, str, str, str]:
    names = list(PROBE_COLOURS)
    pick = (rng or random.SystemRandom()).sample(names, 4)
    return pick[0], pick[1], pick[2], pick[3]


def score_probe_reply(reply: str, expected: tuple[str, str, str, str]) -> int:
    """How many of the four quadrant colours the reply names in order (extra words ignored)."""
    words = [w for w in re.split(r"[^a-z]+", (reply or "").lower()) if w in PROBE_COLOURS]
    return sum(1 for got, want in zip(words[:4], expected, strict=False) if got == want)


async def vision_probe(provider: Any, *, model: str | None = None, quadrants: tuple[str, str, str, str] | None = None) -> dict[str, Any]:
    """One call with a four-colour PNG: does the endpoint really pass images to ``model``?

    Supported only when at least :data:`PROBE_MIN_CORRECT` of the four colours come back in order — a
    text-only model that never saw the image cannot do that, whatever it guesses.
    """
    expected = quadrants or probe_quadrants()
    messages = [{"role": "user", "content": [{"type": "text", "text": VISION_PROBE_PROMPT}, image_part(probe_png(quadrants=expected), "image/png")]}]
    try:
        response = await provider.chat_with_retry(messages, model=model, max_tokens=32, temperature=0.0, retry_mode="standard")
    except Exception as exc:  # a transport that refuses the shape outright
        return {"supported": False, "model": model, "reply": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
    reply = (response.content or "").strip()
    if response.finish_reason == "error" or not reply:
        return {"supported": False, "model": model, "reply": reply[:80] or None, "error": (response.content or "no reply")[:300]}
    correct = score_probe_reply(reply, expected)
    supported = correct >= PROBE_MIN_CORRECT
    return {
        "supported": supported, "model": model, "reply": reply[:80], "expected": list(expected), "correct": correct,
        "error": None if supported else f"named {correct}/4 quadrant colours; the image is not reaching the model",
    }


def cache_path(input_dir: Path) -> Path:
    return Path(input_dir) / FIGURES_CACHE_NAME


def load_cache(input_dir: Path, raw_sha256: str) -> dict[str, Any] | None:
    path = cache_path(input_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if data.get("raw_sha256") == raw_sha256 else None


async def describe_figures(
    provider: Any,
    *,
    paper_dir: Path,
    text: str,
    model: str,
    max_figures: int = MAX_FIGURES,
    max_image_bytes: int = MAX_IMAGE_BYTES,
) -> dict[str, Any]:
    """One call per readable figure; returns the record (descriptions keyed by reference, skips, failures, usage)."""
    figures = find_figures(text)
    descriptions: dict[str, str] = {}
    # ``model`` is the vision model (run.json.figures_model), passed per call: the run's phase model stays pinned
    skipped: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    for figure in figures[:max_figures]:
        path = Path(paper_dir) / figure.ref
        if not path.is_file():
            skipped.append({"ref": figure.ref, "reason": "missing"})
            continue
        data = path.read_bytes()
        mime = image_mime(data)
        if mime is None:
            reason = "lfs_pointer" if data.startswith(b"version https://git-lfs") else "not_an_image"
            skipped.append({"ref": figure.ref, "reason": reason})
            continue
        if len(data) > max_image_bytes:
            skipped.append({"ref": figure.ref, "reason": f"too_large:{len(data)}"})
            continue
        prompt = PROMPT.format(caption=figure.caption or "(no caption)")
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, image_part(data, mime)]}]
        response = await provider.chat_with_retry(messages, model=model, max_tokens=DESCRIPTION_MAX_TOKENS, temperature=0.0, retry_mode="standard")
        for key in usage:
            usage[key] += int((response.usage or {}).get(key, 0) or 0)
        body = (response.content or "").strip()
        if response.finish_reason == "error" or not body:
            failed.append({"ref": figure.ref, "reason": (response.content or "empty reply")[:300]})
            continue
        descriptions[figure.ref] = body
    for figure in figures[max_figures:]:
        skipped.append({"ref": figure.ref, "reason": f"beyond max_figures={max_figures}"})
    return {
        "model": model,
        "figures_found": len(figures),
        "described": len(descriptions),
        "descriptions": descriptions,
        "skipped": skipped,
        "failed": failed,
        "usage": usage,
    }


def apply(input_dir: Path, raw: bytes, record: dict[str, Any]) -> dict[str, Any]:
    """Write ``paper.raw.md`` (benchmark bytes), the enriched ``paper.md`` and the cache; returns what the phase records."""
    input_dir = Path(input_dir)
    raw_sha = hashlib.sha256(raw).hexdigest()
    enriched = insert_descriptions(raw.decode("utf-8", errors="replace"), record.get("descriptions") or {}).encode("utf-8")
    (input_dir / RAW_INPUT_NAME).write_bytes(raw)
    (input_dir / "paper.md").write_bytes(enriched)
    cache = {**record, "raw_sha256": raw_sha, "enriched_sha256": hashlib.sha256(enriched).hexdigest()}
    cache_path(input_dir).write_text(json.dumps(cache, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {k: v for k, v in cache.items() if k != "descriptions"}


__all__ = [
    "BLOCK_START",
    "FIGURES_CACHE_NAME",
    "MAX_FIGURES",
    "PROBE_COLOURS",
    "PROBE_MIN_CORRECT",
    "PROMPT",
    "RAW_INPUT_NAME",
    "VISION_PROBE_PROMPT",
    "Figure",
    "apply",
    "describe_figures",
    "find_figures",
    "image_mime",
    "image_part",
    "insert_descriptions",
    "load_cache",
    "probe_png",
    "probe_quadrants",
    "redact_images",
    "score_probe_reply",
    "vision_probe",
]
