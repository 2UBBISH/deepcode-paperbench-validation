"""Supervised fine-tuning with LoRA -- the upper-bound baseline (Section 4.1).

Hyper-parameters of Appendix F.2 / Table 8: LoRA dropout 0.1, 3 epochs,
learning rate 2e-4, weight decay 0.001, batch size 8 per GPU, max gradient norm
0.3, paged AdamW 32bit and a cosine schedule.  The LoRA rank is chosen so that
the number of trainable parameters matches the corresponding BBOX-ADAPTER
version: ``r = 128`` for the 0.1B adapter and ``r = 384`` for the 0.3B adapter,
with ``alpha = 2r``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..data.prompts import build_generator_prompt
from ..data.loaders import QAExample


@dataclass
class LoRASFTConfig:
    model_name: str = "mistralai/Mixtral-8x7B-v0.1"
    output_dir: str = "runs/sft_lora"
    lora_r: int = 128          # 128 -> ~0.1B adapter, 384 -> ~0.3B adapter
    lora_alpha: Optional[int] = None   # defaults to 2 * r
    lora_dropout: float = 0.1
    num_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 0.3
    optimizer: str = "paged_adamw_32bit"
    lr_scheduler: str = "cosine"
    max_length: int = 512
    target_modules: Optional[List[str]] = None
    seed: int = 0


def build_training_text(dataset: str, example: QAExample) -> str:
    """Prompt + ground-truth solution, i.e. the SFT target."""

    prompt = build_generator_prompt(dataset, example.question, example.choices)
    solution = example.solution or example.answer
    return f"{prompt}\n{solution}"


def run_lora_sft(dataset: str, examples: Sequence[QAExample], config: LoRASFTConfig,
                 tokenizer=None, model=None):
    """Fine-tune Mixtral-8x7B with LoRA.

    This function requires the GPU toolchain (``peft``, ``bitsandbytes``) and is
    therefore never executed in the CPU-only reproduction environment; the code
    mirrors the recipe reported in the paper.
    """

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    os.makedirs(config.output_dir, exist_ok=True)
    alpha = config.lora_alpha or 2 * config.lora_r
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[sft-lora] trainable parameters: {trainable / 1e9:.3f}B")

    texts = [build_training_text(dataset, example) for example in examples]
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=config.max_length,
        padding="max_length",
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    num_steps = max(
        1,
        len(texts) * config.num_epochs
        // (config.per_device_batch_size * config.gradient_accumulation_steps),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    model.train()
    step = 0
    for epoch in range(config.num_epochs):
        for start in range(0, len(texts), config.per_device_batch_size):
            batch = {
                key: value[start:start + config.per_device_batch_size].to(model.device)
                for key, value in encoded.items()
            }
            batch["labels"] = labels[start:start + config.per_device_batch_size].to(model.device)
            loss = model(**batch).loss / config.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 10 == 0:
                print(f"[sft-lora] epoch {epoch} step {step} loss {loss.item():.4f}")
            step += 1

    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    with open(os.path.join(config.output_dir, "sft_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config.__dict__, handle, indent=2)
    return model
