"""Uniform sampling baseline.

"For this baseline, we randomly select partial data from full data to
construct a coreset." (Appendix D.1)
"""

from __future__ import annotations

from typing import Optional

import torch


def uniform_select(n_or_labels, k: int, seed: int = 0,
                   generator: Optional[torch.Generator] = None,
                   **_) -> torch.Tensor:
    """Randomly select ``k`` example indices out of ``n``."""
    if isinstance(n_or_labels, int):
        n = n_or_labels
    else:
        n = int(torch.as_tensor(n_or_labels).numel())
    if generator is None:
        generator = torch.Generator().manual_seed(seed)
    k = min(k, n)
    return torch.randperm(n, generator=generator)[:k]
