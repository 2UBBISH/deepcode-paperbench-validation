"""Shared training / evaluation helpers for the baseline methods."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from apt.trainer import TimeToAccuracy, peak_memory_mb, reset_peak_memory, set_seed
from apt.measure import measure_inference  # noqa: F401  (re-exported for convenience)


def count_parameters(model: nn.Module) -> int:
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def trainable_named_parameters(model: nn.Module):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def batches(features, batch_size: int, shuffle: bool, device, collate: Callable, seed: int = 0):
    indices = list(range(len(features)))
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(indices), generator=g).tolist()
    for i in range(0, len(indices), batch_size):
        chunk = [features[j] for j in indices[i : i + batch_size]]
        batch = collate(chunk)
        yield {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_supervised(
    model: nn.Module,
    task,
    train_features,
    device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    warmup_fraction: float = 0.06,
    trainable: Optional[Iterable[nn.Parameter]] = None,
    eval_features=None,
    raw_eval=None,
    eval_interval: int = 200,
    tta_target: Optional[float] = None,
    eval_batch_size: int = 128,
    seed: int = 42,
    log: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """A plain supervised fine-tuning loop shared by FT / LoRA / retraining."""
    set_seed(seed)
    params = list(trainable) if trainable is not None else [p for p in model.parameters() if p.requires_grad]
    params = [p for p in params if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_features) / batch_size))
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(total_steps * warmup_fraction))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup))
    )

    tta = TimeToAccuracy(tta_target, higher_is_better=task.higher_is_better())
    reset_peak_memory()
    t0 = time.time()
    step = 0
    history: List[Dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        running, n = 0.0, 0
        for batch in batches(train_features, batch_size, True, device, task.collate, seed=seed + epoch):
            opt.zero_grad(set_to_none=True)
            out = task.forward(model, batch, output_hidden_states=False)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            opt.step()
            sched.step()
            running += float(loss.detach().item())
            n += 1
            step += 1
            if eval_features is not None and step % eval_interval == 0:
                metrics = evaluate_task(model, task, eval_features, raw_eval, eval_batch_size, device)
                secs = time.time() - t0
                tta.update(secs, task.primary_metric(metrics))
                rec = {"step": step, "seconds": secs, **metrics}
                history.append(rec)
                if log:
                    log(rec)
        if log:
            log({"epoch": epoch, "loss": running / max(1, n)})

    tta.resolve_with_final()
    result = {
        "wall_time_s": time.time() - t0,
        "tta_s": tta.tta,
        "tta_history": tta.history,
        "peak_train_memory_mb": peak_memory_mb(),
        "n_parameters": count_parameters(model),
        "history": history,
    }
    if eval_features is not None:
        result["final_metrics"] = evaluate_task(model, task, eval_features, raw_eval, eval_batch_size, device)
    return result


@torch.no_grad()
def evaluate_task(model, task, features, raw_eval, batch_size: int, device) -> Dict[str, float]:
    if raw_eval is not None and hasattr(task, "evaluate"):
        return task.evaluate(model, features, raw_eval, batch_size=batch_size, device=str(device))
    model.eval()
    preds, labels = [], []
    for batch in batches(features, batch_size, False, device, task.collate):
        out = task.forward(model, batch, output_hidden_states=False)
        logits = out["logits"]
        if isinstance(logits, tuple):
            logits = logits[0]
        preds.append(logits.detach().cpu())
        labels.append(batch["labels"].detach().cpu())
    model.train()
    if not preds:
        return {task.metric_name: 0.0}
    return task.metrics(torch.cat(preds, 0), {"labels": torch.cat(labels, 0)})


class LogWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def __call__(self, record: Dict[str, Any]) -> None:
        import json

        with open(self.path, "a") as fh:
            fh.write(json.dumps(record) + "\n")


__all__ = [
    "count_parameters",
    "trainable_named_parameters",
    "batches",
    "train_supervised",
    "evaluate_task",
    "measure_inference",
    "LogWriter",
]
