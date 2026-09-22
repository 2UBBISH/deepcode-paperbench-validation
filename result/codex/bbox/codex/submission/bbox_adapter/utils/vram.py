"""Peak GPU memory measurement used for Table 6."""

from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional


@dataclass
class VramReading:
    peak_allocated_gib: float
    peak_reserved_gib: float
    device_index: int = 0


def reset_peak_stats(device: Optional[int] = None) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)


def read_peak(device: Optional[int] = None) -> VramReading:
    import torch

    if not torch.cuda.is_available():
        return VramReading(0.0, 0.0, device or 0)
    gig = 1024 ** 3
    return VramReading(
        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / gig,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device) / gig,
        device_index=device or 0,
    )


@contextmanager
def measure_peak(device: Optional[int] = None):
    """Context manager measuring the peak VRAM of the enclosed block."""

    reset_peak_stats(device)
    reading = VramReading(0.0, 0.0, device or 0)
    try:
        yield reading
    finally:
        final = read_peak(device)
        reading.peak_allocated_gib = final.peak_allocated_gib
        reading.peak_reserved_gib = final.peak_reserved_gib
