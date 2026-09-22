"""Fixing errors in a PTLM: head-only, LoRA and full fine-tuning.

Sec. 4.1: "We fine-tune the entire model (Full FT), low-rank learnable weights
(LoRA), or the LM head only.  We perform 30 steps of parameter updates on a
single online learning example to fix the error when we apply LoRA or full FT,
and 100 steps when we only fine-tune the heads."

LoRA is applied exactly as in the addendum: rank 16, alpha 32, dropout 0.1, no
bias, on the query and value matrices of all self-attention layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ..config import LORA_CONFIG, ExperimentConfig
from ..data.types import Example
from .lm import Seq2SeqLM


# --------------------------------------------------------------------------------------
# parameter selection
# --------------------------------------------------------------------------------------
def apply_lora(model: Seq2SeqLM, **overrides) -> None:
    """Wrap the model with LoRA adapters (query/value matrices, addendum)."""
    from peft import LoraConfig, TaskType, get_peft_model

    kwargs = dict(LORA_CONFIG)
    kwargs.update(overrides)
    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=kwargs["r"],
        lora_alpha=kwargs["lora_alpha"],
        lora_dropout=kwargs["lora_dropout"],
        bias=kwargs["bias"],
        target_modules=kwargs["target_modules"],
    )
    model.model = get_peft_model(model.model, config)


def mark_trainable(model: Seq2SeqLM, mode: str) -> List[str]:
    """Set requires_grad according to the tuning mode; return tunable names.

    * "head"    -- only the LM head (lm_head for BART and T5).
    * "lora"    -- only the LoRA adapter weights.
    * "full_ft" -- every parameter.
    """
    if mode not in ("head", "lora", "full_ft"):
        raise ValueError(f"unknown tuning mode {mode!r}")
    names: List[str] = []
    for name, param in model.model.named_parameters():
        is_head = name.startswith(model.spec.head_only_target) or ".lm_head." in name
        is_lora = "lora_" in name
        if mode == "head":
            param.requires_grad_(is_head)
        elif mode == "lora":
            param.requires_grad_(is_lora)
        else:
            param.requires_grad_(True)
        if param.requires_grad:
            names.append(name)
    if not names:
        raise RuntimeError(f"no trainable parameters selected for mode {mode!r}")
    return names


def mark_frozen(model: Seq2SeqLM) -> None:
    for param in model.model.parameters():
        param.requires_grad_(False)


def trainable_parameters(model: Seq2SeqLM, mode: str, lr: float) -> List[Dict]:
    """Parameter groups for the refinement optimizer."""
    mark_trainable(model, mode)
    params = [p for p in model.model.parameters() if p.requires_grad]
    return [{"params": params, "lr": lr}]


def snapshot_parameters(model: Seq2SeqLM) -> Dict[str, object]:
    return model.snapshot()


def restore_parameters(model: Seq2SeqLM, snapshot: Dict[str, object]) -> None:
    model.restore(snapshot)


# --------------------------------------------------------------------------------------
# distillation replay (Sec. 4.2)
# --------------------------------------------------------------------------------------
@dataclass
class ReplayBatch:
    """A mini-batch of upstream examples with the base model's cached logits."""

    inputs: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    logits: Optional[np.ndarray] = None   # [B, T, C] teacher-forced logits of f_0
    #: vocabulary ids of the C columns of ``logits``; None means the full
    #: vocabulary, which is only tractable for tiny models (smoke tests).
    vocab_ids: Optional[np.ndarray] = None


def distillation_loss(model: Seq2SeqLM, batch: ReplayBatch, weight: float = 1.0):
    """MSE between the current logits and the cached logits of the base PTLM.

    This is the "distillation loss against the outputs of the base PTLM" of
    Sec. 4.2 (Buzzega et al., 2020a: dark experience replay stores pre-softmax
    logits and matches them, which is what we do here).
    """
    import torch

    if batch.logits is None or not batch.inputs:
        raise ValueError("replay batch needs cached base logits and inputs")
    enc = model._encode_inputs(batch.inputs).to(model.device)
    labels = model.tokenizer(
        batch.targets,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model.max_target_len,
    ).input_ids.to(model.device)
    out = model.model(**enc, labels=labels)
    logits = out.logits
    if batch.vocab_ids is not None:
        # The cached distillation targets are restricted to the top-k candidate
        # vocabulary of D_PT (Sec. 3.2, "we only cache top k = 100 largest
        # logits"), so the logits of the current model are sliced the same way.
        index = torch.as_tensor(np.asarray(batch.vocab_ids), device=logits.device, dtype=torch.long)
        logits = logits.index_select(2, index)
    base = torch.as_tensor(batch.logits, device=logits.device, dtype=logits.dtype)
    length = min(logits.shape[1], base.shape[1])
    loss = torch.nn.functional.mse_loss(logits[:, :length], base[:, :length])
    return weight * loss


# --------------------------------------------------------------------------------------
# the actual update
# --------------------------------------------------------------------------------------
def fix_single_error(
    model: Seq2SeqLM,
    example: Example,
    *,
    steps: int,
    lr: float,
    mode: str,
    replay_pool: Optional[Callable[[int], ReplayBatch]] = None,
    replay_every_n_steps: int = 10,
    distill_weight: float = 1.0,
    optimizer: Optional[object] = None,
    return_losses: bool = False,
):
    """Fine-tune model (in place) to fix a single error.

    replay_pool(step) -> ReplayBatch is called every replay_every_n_steps steps
    and its distillation loss is added to the refinement loss (Sec. 4.2: "we
    sparsely replay a mini-batch of 8 examples every 10 training steps").
    """
    import torch

    params = trainable_parameters(model, mode, lr)
    if optimizer is None:
        optimizer = torch.optim.AdamW(params, lr=lr)
    model.train()
    losses: List[float] = []
    for step in range(1, steps + 1):
        enc = model._encode_inputs([example.input]).to(model.device)
        labels = model.tokenizer(
            [example.target],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.max_target_len,
        ).input_ids.to(model.device)
        out = model.model(**enc, labels=labels)
        loss = out.loss
        if replay_pool is not None and replay_every_n_steps > 0 and step % replay_every_n_steps == 0:
            batch = replay_pool(step) if callable(replay_pool) else replay_pool
            if batch is not None and batch.inputs:
                loss = loss + distillation_loss(model, batch, weight=distill_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    return losses if return_losses else None


def refinement_optimizer(model: Seq2SeqLM, cfg: ExperimentConfig, sequential: bool = False):
    """AdamW over the parameters selected by cfg.tuning_mode (Sec. 4.1)."""
    import torch

    lr = cfg.lr_sequential() if sequential else cfg.lr_single()
    params = trainable_parameters(model, cfg.tuning_mode, lr)
    return torch.optim.AdamW(params, lr=lr)


def prepare_model(cfg: ExperimentConfig, lora: bool = True) -> Seq2SeqLM:
    """Instantiate the base PTLM and make the requested parameters trainable."""
    model = Seq2SeqLM.from_config(cfg)
    if cfg.tuning_mode == "lora" and lora:
        apply_lora(model)
    mark_trainable(model, cfg.tuning_mode)
    return model
