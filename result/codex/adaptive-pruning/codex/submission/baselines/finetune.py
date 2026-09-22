"""Full fine-tuning baseline (the ``FT`` row of Tables 2/3/8)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from apt.trainer import resolve_device

from .common import LogWriter, measure_inference, train_supervised


def run_finetune(
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
        p.requires_grad_(True)
    out_dir = output_dir or config.output_dir
    log = LogWriter(f"{out_dir}/ft_log.jsonl")
    result = train_supervised(
        model,
        task,
        train_features,
        device,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        eval_features=eval_features,
        raw_eval=raw_eval,
        eval_interval=config.eval_interval,
        tta_target=config.tta_target,
        eval_batch_size=config.eval_batch_size,
        seed=config.seed,
        log=log,
    )
    result.update({"method": "FT", "sparsity": 0.0})
    if measure and eval_features is not None:
        result["inference"] = measure_inference(
            model, task, eval_features, device, batch_size=config.eval_batch_size
        )
    return result


__all__ = ["run_finetune"]
