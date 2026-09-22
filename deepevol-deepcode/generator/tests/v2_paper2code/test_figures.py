"""Figures described from the image and put back after their reference (intake's optional pass)."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

from apps.v2.agent.paper2code import figures
from apps.v2.agent_engine.paper2code.seams.llm_runtime import LLMResponse
from tests.v2_paper2code.fake_vision import answer_probe, decode_quadrants

TEXT = "# T\n\nintro\n\n![](assets/asset_1.jpg)\n\nFigure 1. PPO vs batch size.\n\n![](assets/asset_2.jpg)\nFigure 2. Arch.\n\ntail\n"


class VisionProvider:
    """``supported``: sees the image; ``blind``: a text model whose endpoint dropped the image part (guesses)."""

    def __init__(self, *, supported: bool = True, blind: bool = False) -> None:
        self.supported = supported
        self.blind = blind
        self.calls: list[list[dict[str, Any]]] = []
        self.models: list[str | None] = []

    async def chat_with_retry(self, messages, **kwargs) -> LLMResponse:
        self.calls.append(messages)
        self.models.append(kwargs.get("model"))
        parts = messages[-1]["content"]
        text = next(p["text"] for p in parts if p.get("type") == "text")
        if not self.supported:
            return LLMResponse(content="image input is not supported for this model", finish_reason="error", usage={})
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 0}
        if text.startswith(figures.VISION_PROBE_PROMPT):
            return LLMResponse(content="Red, green, blue, yellow." if self.blind else answer_probe(messages) + ".", usage=usage)
        caption = text.split("\n\n")[1]
        return LLMResponse(content=f"Line plot for: {caption}\nx: batch size 1k..32k", usage=usage)


def _image_url(messages: list[dict[str, Any]]) -> str:
    return next(p["image_url"]["url"] for p in messages[-1]["content"] if p.get("type") == "image_url")


def test_find_figures_and_captions() -> None:
    found = figures.find_figures(TEXT)
    assert [(f.ref, f.caption) for f in found] == [("assets/asset_1.jpg", "Figure 1. PPO vs batch size."), ("assets/asset_2.jpg", "Figure 2. Arch.")]
    assert found[0].line_no == 4


def test_insert_is_in_place_marked_and_idempotent() -> None:
    once = figures.insert_descriptions(TEXT, {"assets/asset_1.jpg": "Line plot.\n\nx: batch size"})
    lines = once.splitlines()
    i = lines.index("![](assets/asset_1.jpg)")
    assert lines[i + 1] == ""
    assert lines[i + 2] == figures.BLOCK_START
    assert lines[i + 3] == ">"
    assert lines[i + 4] == "> Line plot."
    assert lines[i + 5] == ">"
    assert lines[i + 6] == "> x: batch size"
    assert "Figure 1. PPO vs batch size." in once
    assert once.endswith("tail\n")
    assert "![](assets/asset_2.jpg)\nFigure 2. Arch." in once  # undescribed reference untouched
    assert figures.insert_descriptions(once, {"assets/asset_1.jpg": "other"}) == once


def test_probe_png_and_vision_probe() -> None:
    quadrants = ("purple", "black", "orange", "white")
    png = figures.probe_png(quadrants=quadrants)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert figures.image_mime(png) == "image/png"
    assert decode_quadrants(png) == quadrants
    seeing = VisionProvider()
    ok = asyncio.run(figures.vision_probe(seeing, model="vision-model"))
    assert ok["supported"] is True
    assert ok["correct"] == 4
    assert seeing.models == ["vision-model"]
    no = asyncio.run(figures.vision_probe(VisionProvider(supported=False)))
    assert no["supported"] is False
    assert "not supported" in no["error"]
    # a text model whose endpoint silently drops the image guesses colours: 1/8 per quadrant, refused
    blind = asyncio.run(figures.vision_probe(VisionProvider(blind=True), quadrants=("black", "white", "purple", "orange")))
    assert blind["supported"] is False
    assert blind["correct"] == 0
    assert "not reaching the model" in blind["error"]
    assert figures.score_probe_reply("Top-left: red; top-right: GREEN, then blue, then purple", ("red", "green", "blue", "yellow")) == 3


def test_describe_skips_pointers_and_missing_and_describes_images(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "asset_1.jpg").write_bytes(b"\xff\xd8\xff\xe0jpegbytes")
    (tmp_path / "assets" / "asset_2.jpg").write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:x\nsize 1\n")
    text = TEXT + "\n![](assets/asset_3.jpg)\nFigure 3. Missing.\n"
    provider = VisionProvider()
    record = asyncio.run(figures.describe_figures(provider, paper_dir=tmp_path, text=text, model="m"))
    assert provider.models == ["m"]
    assert record["figures_found"] == 3
    assert record["described"] == 1
    assert record["skipped"] == [{"ref": "assets/asset_2.jpg", "reason": "lfs_pointer"}, {"ref": "assets/asset_3.jpg", "reason": "missing"}]
    assert record["descriptions"]["assets/asset_1.jpg"].startswith("Line plot for: Figure 1. PPO vs batch size.")
    assert record["usage"] == {"prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 0}
    url = _image_url(provider.calls[0])
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\xff\xd8\xff\xe0jpegbytes"


def test_apply_keeps_raw_bytes_and_caches(tmp_path: Path) -> None:
    raw = TEXT.encode()
    record = {"model": "m", "figures_found": 2, "described": 1, "descriptions": {"assets/asset_1.jpg": "Line plot."}, "skipped": [], "failed": [], "usage": {}}
    summary = figures.apply(tmp_path, raw, record)
    assert (tmp_path / "paper.raw.md").read_bytes() == raw
    enriched = (tmp_path / "paper.md").read_text()
    assert figures.BLOCK_START in enriched
    assert "descriptions" not in summary
    cache = json.loads((tmp_path / "figures.json").read_text())
    assert cache["descriptions"] == record["descriptions"]
    assert figures.load_cache(tmp_path, summary["raw_sha256"]) is not None
    assert figures.load_cache(tmp_path, "0" * 64) is None


def test_redact_images_for_the_call_log() -> None:
    messages = [{"role": "user", "content": [{"type": "text", "text": "t"}, figures.image_part(b"\x89PNG\r\n\x1a\nxx", "image/png")]}]
    redacted = figures.redact_images(messages)
    assert redacted[0]["content"][0] == {"type": "text", "text": "t"}
    assert redacted[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,<")
    assert "redacted" in redacted[0]["content"][1]["image_url"]["url"]
    assert "base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nxx").decode() in messages[0]["content"][1]["image_url"]["url"]  # original untouched
    assert figures.redact_images([{"role": "user", "content": "plain"}]) == [{"role": "user", "content": "plain"}]
