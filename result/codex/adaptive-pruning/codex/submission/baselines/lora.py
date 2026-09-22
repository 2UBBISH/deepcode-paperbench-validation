"""Standard LoRA baseline (``LoRA`` rows of Tables 2 and 3).

Low-rank adapters are added next to the frozen query / value projections
(optionally the FFN projections for the smaller models) and are merged back
into the frozen weights after training.  Because the LM is *not* pruned its
inference cost equals the dense model's, which is exactly the limitation the
paper motivates APT with.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from apt.trainer import resolve_device

from .common import LogWriter, measure_inference, train_supervised


class LoRALinear(nn.Module):
    """``H = W X + s B A X`` with a frozen ``W``."""

    def __init__(self, base: nn.Linear, r: int = 8, scaling: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = scaling
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scaling * F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)

    @torch.no_grad()
    def merge(self) -> None:
        self.base.weight.add_(self.scaling * (self.lora_B @ self.lora_A))


def attach_lora(
    model: nn.Module,
    r: int = 8,
    scaling: float = 2.0,
    targets: Optional[List[str]] = None,
) -> List[Tuple[nn.Module, str, LoRALinear]]:
    """Replace the target projections by :class:`LoRALinear` wrappers.

    Returns ``(parent, attribute, wrapper)`` triples so that the adapters can be
    merged and removed again by :func:`merge_and_unwrap`.
    """
    # RoBERTa/BERT name the projections ``query``/``value``; T5 uses ``q``/``v``.
    targets = targets or ["query", "value", "q", "v"]
    wrapped: List[Tuple[nn.Module, str, LoRALinear]] = []
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                new = LoRALinear(child, r=r, scaling=scaling)
                setattr(module, name, new)
                wrapped.append((module, name, new))
    return wrapped


def merge_and_unwrap(wrapped: List[Tuple[nn.Module, str, LoRALinear]]) -> None:
    """Fold every adapter into its base weight and restore a plain ``nn.Linear``.

    Used by the ``LoRA+Prune`` baseline, which applies Mask Tuning to the
    *merged* LoRA-tuned model.
    """
    for parent, attr, lin in wrapped:
        lin.merge()
        setattr(parent, attr, lin.base)


def run_lora(
    model,
    tokenizer,
    task,
    config,
    train_features,
    eval_features=None,
    raw_eval=None,
    output_dir: Optional[str] = None,
    measure: bool = True,
) -> Dict[str, Any]:
    device = resolve_device(config.device)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = attach_lora(model, r=config.initial_rank, scaling=config.scaling)
    params = [p for _, _, lin in wrapped for p in (lin.lora_A, lin.lora_B)]

    out_dir = output_dir or config.output_dir
    log = LogWriter(f"{out_dir}/lora_log.jsonl")
    result = train_supervised(
        model,
        task,
        train_features,
        device,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        trainable=params,
        eval_features=eval_features,
        raw_eval=raw_eval,
        eval_interval=config.eval_interval,
        tta_target=config.tta_target,
        eval_batch_size=config.eval_batch_size,
        seed=config.seed,
        log=log,
    )
    result.update({"method": "LoRA", "sparsity": 0.0})
    if measure and eval_features is not None:
        result["inference"] = measure_inference(
            model, task, eval_features, device, batch_size=config.eval_batch_size
        )
    return result


__all__ = ["LoRALinear", "attach_lora", "merge_and_unwrap", "run_lora"]
