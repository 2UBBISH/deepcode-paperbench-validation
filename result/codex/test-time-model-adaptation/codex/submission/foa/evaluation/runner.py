"""Generic online evaluation loop for test-time adaptation methods.

The paper evaluates every method on a *stream* of test samples (ImageNet-C severities,
ImageNet-R/V2/Sketch, and the non-i.i.d. variants of Section 4.4).  The runner below
implements that loop once and works for every method, including the interval-based
``FOA-I`` for which predictions are deferred until an interval is complete.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

from .metrics import OnlineMetrics


@dataclass
class StreamResult:
    acc: float
    ece: float
    num_samples: int
    seconds: float
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"acc": self.acc, "ece": self.ece, "num_samples": self.num_samples}
        d.update(self.extra)
        return d


def run_stream(
    method,
    stream: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    max_samples: Optional[int] = None,
    n_bins: int = 15,
    progress: bool = False,
    log_every: Optional[int] = None,
    collect_extra: bool = True,
) -> StreamResult:
    """Run ``method`` over ``stream`` and return the accuracy / ECE of the whole stream.

    Args:
        method: object with ``reset`` / ``step`` / ``flush`` (see
            :class:`foa.methods.base.TTAMethod`).
        stream: iterable of ``(images, targets)`` batches.
        max_samples: stop after this many samples (useful for debugging).
    """
    if hasattr(method, "reset"):
        method.reset()
    metrics = OnlineMetrics(n_bins=n_bins)
    pending: List[torch.Tensor] = []
    extras: Dict[str, List[float]] = {}
    start = time.time()

    iterator = stream
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(stream)  # type: ignore[assignment]
        except ImportError:  # pragma: no cover
            iterator = stream

    def consume(logits: torch.Tensor) -> None:
        nonlocal pending
        if logits is None or logits.numel() == 0:
            return
        all_targets = torch.cat(pending, 0) if pending else torch.empty(0, dtype=torch.long)
        n = int(logits.shape[0])
        if n > all_targets.numel():  # pragma: no cover - defensive
            n = int(all_targets.numel())
        metrics.update(logits[:n], all_targets[:n])
        rest = all_targets[n:]
        pending = [rest] if rest.numel() else []

    for images, targets in iterator:
        images = images if torch.is_tensor(images) else torch.as_tensor(images)
        targets = targets if torch.is_tensor(targets) else torch.as_tensor(targets)
        pending.append(targets)
        out = method.step(images)
        if out is not None:
            consume(out)
        if collect_extra and getattr(method, "last_extra", None):
            for k, v in method.last_extra.items():
                extras.setdefault(k, []).append(float(v))
        if log_every and metrics.num_samples and metrics.num_samples % log_every < images.shape[0]:
            cur = metrics.compute()
            print(
                f"[runner] {metrics.num_samples} samples | acc={cur['acc']:.2f} "
                f"ece={cur['ece']:.2f}",
                flush=True,
            )
        if max_samples is not None and metrics.num_samples >= max_samples:
            break

    if pending:
        out = method.flush()
        if out is not None:
            consume(out)

    elapsed = time.time() - start
    result = metrics.compute()
    extra = {k: float(sum(v) / len(v)) for k, v in extras.items() if v}
    return StreamResult(
        acc=result["acc"],
        ece=result["ece"],
        num_samples=result["num_samples"],
        seconds=elapsed,
        extra=extra,
    )
