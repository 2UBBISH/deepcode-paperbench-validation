#!/usr/bin/env python
"""GPU memory measurements of Table 6 (Mixtral-8x7B, 0.1B adapter).

The table reports peak VRAM for

* the base model (half precision)                          -- inference only,
* base + LoRA fine-tuning (supervised fine-tuning upper bound) -- SFT-LoRA r=128,
* base + BBOX-ADAPTER (BERT-0.1B backend)                  -- adapter training and
  adapted inference.

Accuracy columns of the same table are produced by ``run_cot_baseline.py`` (base
model), ``run_sft_lora.py`` (base + LoRA) and ``train_bbox_adapter.py``
(base + BBOX-ADAPTER, ``--llm-provider huggingface``).
"""

from __future__ import annotations

import os

from _common import base_parser, build_config, write_json

from bbox_adapter.data.loaders import load_dataset_examples
from bbox_adapter.inference.adaptive_inference import AdaptedInference
from bbox_adapter.pipeline import build_adapter
from bbox_adapter.utils.vram import measure_peak


def _load_mixtral(model_name: str, dtype: str = "float16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, dtype), device_map="auto"
    )
    model.eval()
    return model, tokenizer


def measure_base(model_name: str) -> float:
    import torch

    with measure_peak() as reading:
        model, _ = _load_mixtral(model_name)
        inputs = None
        del model
    return reading.peak_allocated_gib


def measure_lora(model_name: str, r: int = 128) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model  # type: ignore

    results = {}
    with measure_peak() as reading:
        model, tokenizer = _load_mixtral(model_name)
        config = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.1, bias="none",
                            task_type="CAUSAL_LM",
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, config)
        model.train()
        batch = tokenizer(["Q: What is 1 + 1?\nA:"], return_tensors="pt").to(model.device)
        out = model(**batch, labels=batch["input_ids"])
        out.loss.backward()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        optimizer.step()
        del optimizer
    results["train_vram_gib"] = reading.peak_allocated_gib

    with measure_peak() as reading:
        model.eval()
        with torch.no_grad():
            model.generate(**batch, max_new_tokens=8)
    results["inference_vram_gib"] = reading.peak_allocated_gib
    return results


def measure_bbox_adapter(config, dataset: str, size: str = "0.1B") -> dict:
    """Base model + BBOX-ADAPTER training / inference memory."""

    import torch

    results = {}
    with measure_peak() as reading:
        model, tokenizer = _load_mixtral(config.llm.name)
        adapter = build_adapter(config, dataset, size)
        # Adapter training: score the model's own generations (Eq. 3).
        from bbox_adapter.adapter.trainer import TrainingExample

        examples = [
            TrainingExample(
                key="probe",
                question="Is the sky blue?",
                positive="The sky is blue during the day.",
                negatives=["The sky is green at night."],
            )
        ]
        adapter.fit(examples, num_steps=1)
        del model
    results["train_vram_gib"] = reading.peak_allocated_gib

    with measure_peak() as reading:
        model, tokenizer = _load_mixtral(config.llm.name)
        inference = AdaptedInference(
            llm=None, adapter=adapter, dataset=dataset,
            beam_size=config.inference.beam_size,
        )
        inference.adapter.score_batch(
            ["Is the sky blue?"],
            ["The sky is blue during the day."],
        )
        del model
    results["inference_vram_gib"] = reading.peak_allocated_gib
    return results


def main() -> None:
    parser = base_parser("Measure VRAM for Table 6.")
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--lora-r", type=int, default=128)
    args = parser.parse_args()

    config = build_config(args)
    config.llm.name = args.model
    config.llm.provider = "huggingface"
    dataset = (args.dataset or config.dataset).lower()
    os.makedirs(config.output_dir, exist_ok=True)

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("VRAM measurements require a GPU")
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(str(exc))

    payload = {
        "base_model_inference_vram_gib": measure_base(args.model),
        "lora": measure_lora(args.model, args.lora_r),
        "bbox_adapter": measure_bbox_adapter(config, dataset, args.adapter_size or "0.1B"),
    }
    write_json(os.path.join(config.output_dir, "vram_results.json"), payload)
    print(payload)


if __name__ == "__main__":
    main()
