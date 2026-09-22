"""Efficiency measurement (Section 5.3 and the task addendum).

* **Training peak memory** -- ``torch.cuda.max_memory_allocated()``.
* **Training speed** -- time-to-accuracy (TTA): "the time-to-accuracy of
  reaching 97% of the dev (/test) set performance of the finetuning baseline"
  (``apt.trainer.TimeToAccuracy``).
* **Inference speed** -- the *inference throughput*: sampled processed per
  second.
* **Inference peak memory** -- ``torch.cuda.max_memory_allocated()``.

Both training and inference are measured with a single GPU in the paper; the
batch sizes are 128 for the small models (RoBERTa/T5) and 32 / 4 for LLaMA-7B /
13B.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import torch

from .trainer import peak_memory_mb, reset_peak_memory


def _iter_chunks(features, batch_size: int, collate):
    for i in range(0, len(features), batch_size):
        chunk = [features[j] for j in range(i, min(i + batch_size, len(features)))]
        yield collate(chunk)


@torch.no_grad()
def measure_inference(
    model,
    task,
    features,
    device,
    batch_size: int = 128,
    max_batches: int = 20,
    warmup_batches: int = 1,
) -> Dict[str, float]:
    """Inference throughput (samples/s) and peak memory (MB)."""
    model.eval()
    reset_peak_memory()
    n_samples = 0
    elapsed = 0.0
    n_batches = 0
    for i, batch in enumerate(_iter_chunks(features, batch_size, task.collate)):
        if i >= max_batches + warmup_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        if i >= warmup_batches:
            t0 = time.time()
        with torch.inference_mode():
            task.forward(model, batch, output_hidden_states=False)
        if i >= warmup_batches:
            elapsed += time.time() - t0
            n_samples += int(batch["input_ids"].shape[0])
            n_batches += 1
    model.train()
    elapsed = max(1e-6, elapsed)
    return {
        "throughput_samples_per_s": n_samples / elapsed,
        "inference_time_ms": 1000.0 * elapsed / max(1, n_batches),
        "inference_time_ms_per_sample": 1000.0 * elapsed / max(1, n_samples),
        "inference_peak_memory_mb": peak_memory_mb(),
        "batch_size": batch_size,
    }


def relative_efficiency(
    reference: Dict[str, float], other: Dict[str, float]
) -> Dict[str, float]:
    """Normalise ``other`` against the fine-tuning ``reference`` (Tables 2/11)."""
    out: Dict[str, float] = {}
    ref_tp = reference.get("throughput_samples_per_s")
    other_tp = other.get("throughput_samples_per_s")
    if ref_tp and other_tp:
        out["inference_speed_pct"] = 100.0 * ref_tp / other_tp
    ref_mem = reference.get("inference_peak_memory_mb")
    other_mem = other.get("inference_peak_memory_mb")
    if ref_mem and other_mem:
        out["inference_memory_pct"] = 100.0 * other_mem / ref_mem
    return out


def efficiency_report(
    dense_model,
    pruned_model,
    task,
    features,
    device,
    batch_size: int = 128,
    max_batches: int = 20,
) -> Dict[str, Any]:
    """Side-by-side inference efficiency of the dense and the pruned LM."""
    dense = measure_inference(dense_model, task, features, device, batch_size, max_batches)
    pruned = measure_inference(pruned_model, task, features, device, batch_size, max_batches)
    return {
        "dense": dense,
        "pruned": pruned,
        "relative": relative_efficiency(dense, pruned),
        "n_parameters_dense": int(sum(p.numel() for p in dense_model.parameters())),
        "n_parameters_pruned": int(sum(p.numel() for p in pruned_model.parameters())),
    }


__all__ = [
    "measure_inference",
    "relative_efficiency",
    "efficiency_report",
]
