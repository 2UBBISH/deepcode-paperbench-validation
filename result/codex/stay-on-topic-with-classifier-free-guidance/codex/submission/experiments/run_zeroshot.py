#!/usr/bin/env python3
"""Section 3.1 / Table 5 -- CFG on zero-shot benchmarks.

Evaluates the GPT-2 and Pythia model families (the LLaMA family is listed for
completeness but is out of scope for the reproduction) on the nine zero-shot
benchmarks of the LM Evaluation Harness, sweeping the guidance strength
``gamma``.

The output JSON contains, for every (model, task, gamma) triple, the accuracy
of the harness metric plus the inference FLOPs per token of that model at the
sequence length of the task -- everything ``run_flops_ancova.py`` needs for
the accuracy-vs-FLOP analysis of Section 4.

Example
-------
    python experiments/run_zeroshot.py \
        --models gpt2 gpt2-medium gpt2-large gpt2-xl \
        --tasks arc_challenge arc_easy boolq hellaswag piqa sciq triviaqa \
                winogrande lambada_openai \
        --gammas 1.0 1.1 1.25 1.5 1.75 2.0 \
        --output results/zeroshot.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfglm.flops import flops_per_token  # noqa: E402
from cfglm.harness import evaluate_task  # noqa: E402
from cfglm.models import (  # noqa: E402
    GPT2_MODELS,
    PYTHIA_MODELS,
    LLAMA_MODELS,
    resolve_model,
)
from cfglm.tasks import TABLE5_TASKS  # noqa: E402

DEFAULT_GAMMAS = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0]

# Average sequence length (context + continuation) per task, used for the
# attention term of the FLOP estimate.  These follow the harness's typical
# token counts for zero-shot evaluation.
TASK_SEQ_LEN: Dict[str, int] = {
    "arc_challenge": 64,
    "arc_easy": 64,
    "boolq": 256,
    "hellaswag": 256,
    "piqa": 64,
    "sciq": 128,
    "triviaqa": 32,
    "winogrande": 64,
    "lambada_openai": 128,
}


def load_models(keys: List[str], device: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    for key in keys:
        spec = resolve_model(key)
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(spec.hf_name, torch_dtype=torch_dtype)
        model.to(device)
        model.eval()
        yield spec, model, tokenizer


def _extract_metric(task_results: dict, name: str):
    """Pull ``acc`` / ``acc_norm`` out of an lm-eval result dict.

    lm-eval keys its metrics as ``"acc,none"`` (newer versions) or ``"acc"``
    (older ones); both are handled.
    """
    for key in (name, f"{name},none", f"{name},stderr"):
        if key in task_results and not key.endswith("stderr"):
            return float(task_results[key])
    return None


def evaluate_with_lm_eval(
    model_key: str,
    tasks: List[str],
    gamma: float,
    uncond_prefix_tokens: int,
    batch_size: int,
    limit,
) -> Dict[str, dict]:
    """Evaluate a model with the canonical EleutherAI harness.

    This is the exact setup used for Table 5 of the paper: the harness's own
    task definitions and metrics, with CFG applied through
    :class:`cfglm.lm_eval_adapter.CFGHFLM`.
    """
    import lm_eval  # imported lazily: the native backend has no such dependency

    from cfglm.lm_eval_adapter import CFGHFLM, register

    register()
    lm = CFGHFLM(
        pretrained=resolve_model(model_key).hf_name,
        gamma=gamma,
        uncond_prefix_tokens=uncond_prefix_tokens,
        batch_size=batch_size,
    )
    results = lm_eval.simple_evaluate(model=lm, tasks=list(tasks), limit=limit)
    return results["results"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=[m.key for m in GPT2_MODELS] + [m.key for m in PYTHIA_MODELS])
    parser.add_argument("--tasks", nargs="+", default=TABLE5_TASKS)
    parser.add_argument("--gammas", nargs="+", type=float, default=DEFAULT_GAMMAS)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples per task")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--uncond-prefix-tokens", type=int, default=1,
                        help="number of prompt tokens kept as the unconditional context (Sec. 3.1)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16" if torch.cuda.is_available() else "float32",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", default="results/zeroshot.json")
    parser.add_argument(
        "--backend",
        default="native",
        choices=["native", "lm_eval"],
        help="native = cfglm.harness (no extra dependency); "
             "lm_eval = the canonical EleutherAI harness (needs lm-eval installed)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    records: List[dict] = []

    if args.backend == "lm_eval":
        for model_key in args.models:
            spec = resolve_model(model_key)
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(spec.hf_name)
            for gamma in args.gammas:
                results = evaluate_with_lm_eval(
                    model_key,
                    args.tasks,
                    gamma,
                    args.uncond_prefix_tokens,
                    args.batch_size,
                    args.limit,
                )
                for task_name, task_results in results.items():
                    acc = _extract_metric(task_results, "acc")
                    acc_norm = _extract_metric(task_results, "acc_norm")
                    seq_len = TASK_SEQ_LEN.get(task_name, 128)
                    record = {
                        "model": spec.key,
                        "hf_name": spec.hf_name,
                        "family": spec.family,
                        "params": spec.params,
                        "n_params": spec.n_params,
                        "task": task_name,
                        "gamma": gamma,
                        "cfg": gamma != 1.0,
                        "acc": acc,
                        "acc_norm": acc_norm,
                        "substring_match": None,
                        "n": None,
                        "seq_len": seq_len,
                        "flops_per_token": flops_per_token(
                            config, seq_len, n_passes=2 if gamma != 1.0 else 1
                        ),
                        "backend": "lm_eval",
                    }
                    records.append(record)
                    print(f"[zeroshot/lm_eval] {spec.key} | {task_name} | gamma={gamma} | acc={acc}")
                    with open(args.output, "w") as fh:
                        json.dump(records, fh, indent=2)
        print(f"\nWrote {len(records)} records to {args.output}")
        return

    for spec, model, tokenizer in load_models(args.models, args.device, args.dtype):
        print(f"\n=== {spec.key} ({spec.hf_name}) ===")
        for gamma in args.gammas:
            for task_name in args.tasks:
                result = evaluate_task(
                    model,
                    tokenizer,
                    task_name,
                    gamma,
                    limit=args.limit,
                    uncond_prefix_tokens=args.uncond_prefix_tokens,
                    batch_size=args.batch_size,
                )
                seq_len = TASK_SEQ_LEN.get(task_name, 128)
                n_passes = 2 if gamma != 1.0 else 1
                record = {
                    "model": spec.key,
                    "hf_name": spec.hf_name,
                    "family": spec.family,
                    "params": spec.params,
                    "n_params": spec.n_params,
                    "task": task_name,
                    "gamma": gamma,
                    "cfg": gamma != 1.0,
                    "acc": result.metrics.get("acc"),
                    "acc_norm": result.metrics.get("acc_norm"),
                    "substring_match": result.metrics.get("substring_match"),
                    "n": result.n,
                    "seq_len": seq_len,
                    "flops_per_token": flops_per_token(model.config, seq_len, n_passes=n_passes),
                    "backend": "native",
                }
                records.append(record)
                metric = record["acc"] if record["acc"] is not None else record["substring_match"]
                print(
                    f"[zeroshot] {spec.key:>16} | {task_name:<15} | gamma={gamma:<5} "
                    f"| acc={metric if metric is None else round(metric, 4)} | "
                    f"flops/token={record['flops_per_token']:.3e}"
                )
                with open(args.output, "w") as fh:
                    json.dump(records, fh, indent=2)

    print(f"\nWrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
