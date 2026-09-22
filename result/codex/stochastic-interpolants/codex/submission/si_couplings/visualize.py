"""Figures 3-6: in-painting / super-resolution panels and temporal slices.

Every panel is written in the layout used by the paper: for in-painting the
left image is the base sample x_0 (the low-information/masked input), the
middle image is the model sample X_{t=1} and the right image is the ground
truth; for super-resolution the left image is the low-resolution input.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor


def to_uint8(x: Tensor) -> Tensor:
    """[-1, 1] float image -> [0, 255] uint8."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def triplet_grid(left: Tensor, middle: Tensor, right: Tensor, gap: int = 4) -> Tensor:
    """Stack rows of [left | middle | right] image triplets into one grid."""
    rows = []
    for a, b, c in zip(left, middle, right):
        sep_v = torch.zeros(a.shape[0], a.shape[-2], gap)
        rows.append(torch.cat([a, sep_v, b, sep_v, c], dim=-1))
    out = []
    for i, row in enumerate(rows):
        if i:
            out.append(torch.zeros(row.shape[0], gap, row.shape[2]))
        out.append(row)
    return torch.cat(out, dim=1)  # stack rows of images vertically


def save_grid(x: Tensor, path: str | os.PathLike) -> None:
    from torchvision.utils import save_image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(to_uint8(x).float() / 255.0, path)


def inpainting_panel(result: dict, path: str | os.PathLike, max_items: int = 6) -> None:
    """result = output of :func:`si_couplings.sample.sample_inpainting`."""
    n = min(max_items, result["sample"].shape[0])
    grid = triplet_grid(result["x0"][:n], result["sample"][:n], result["x1"][:n])
    save_grid(grid, path)


def super_resolution_panel(result: dict, path: str | os.PathLike, max_items: int = 6) -> None:
    """result = output of :func:`si_couplings.sample.sample_super_resolution`."""
    n = min(max_items, result["sample"].shape[0])
    low = result["low_res"][:n]
    if low.shape[-1] != result["sample"].shape[-1]:
        low = torch.nn.functional.interpolate(
            low, size=result["sample"].shape[-2:], mode="nearest"
        )
    truth = result.get("x1")
    right = truth[:n] if truth is not None else result["sample"][:n]
    grid = triplet_grid(low, result["sample"][:n], right)
    save_grid(grid, path)


@torch.no_grad()
def temporal_slices(
    model,
    x0: Tensor,
    *,
    cond: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    times: Optional[Tensor] = None,
    path: str | os.PathLike = "results/temporal_slices.png",
) -> Tensor:
    """Probability-flow snapshots at a list of times (Figures 5 and 6)."""
    from .sample import sample_from_base

    times = torch.linspace(0, 1, 6) if times is None else times
    frames = []
    for t0, t1 in zip(times[:-1], times[1:]):
        frames.append(sample_from_base(model, x0, cond=cond, labels=labels, mask=mask,
                                       method="dopri5", t0=float(t0), t1=float(t1)))
    stack = torch.stack(frames)  # (T, B, C, H, W)
    grid = torch.cat([stack[t, : min(4, stack.shape[1])] for t in range(stack.shape[0])], dim=0)
    save_grid(grid, path)
    return stack


__all__ = [
    "to_uint8",
    "triplet_grid",
    "save_grid",
    "inpainting_panel",
    "super_resolution_panel",
    "temporal_slices",
]
