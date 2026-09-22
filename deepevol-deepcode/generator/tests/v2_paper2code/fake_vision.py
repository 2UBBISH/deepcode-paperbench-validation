"""A fake that really looks at the probe image: decodes the four-quadrant PNG and names its colours."""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any

from apps.v2.agent.paper2code import figures


def decode_quadrants(png: bytes) -> tuple[str, ...]:
    """The colour names of the four quadrants of a :func:`figures.probe_png` image (stdlib decode, filter 0 only)."""
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", png[16:24])
    pos, idat = 8, b""
    while pos < len(png):
        length, kind = struct.unpack(">I4s", png[pos : pos + 8])
        if kind == b"IDAT":
            idat += png[pos + 8 : pos + 8 + length]
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = 1 + 3 * width

    def pixel(x: int, y: int) -> tuple[int, int, int]:
        off = y * stride + 1 + 3 * x
        return raw[off], raw[off + 1], raw[off + 2]

    by_rgb = {rgb: name for name, rgb in figures.PROBE_COLOURS.items()}
    points = ((1, 1), (width - 2, 1), (1, height - 2), (width - 2, height - 2))
    return tuple(by_rgb[pixel(x, y)] for x, y in points)


def image_bytes(messages: list[dict[str, Any]]) -> bytes:
    url = next(p["image_url"]["url"] for p in messages[-1]["content"] if p.get("type") == "image_url")
    return base64.b64decode(url.split(",", 1)[1])


def answer_probe(messages: list[dict[str, Any]]) -> str:
    return ", ".join(decode_quadrants(image_bytes(messages)))
