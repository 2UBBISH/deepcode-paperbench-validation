#!/usr/bin/env python3
"""Section 3.4 -- negative prompting for assistants (mechanism only).

The human preference study of Section 3.4 is **out of scope** for this
reproduction (the addendum excludes it).  What *is* part of the CFG
framework is the negative-prompting mechanism of Equation 5, which is
implemented in ``cfglm/generation.py`` and exercised here.

The script reproduces the sampling protocol of the experiment: it draws
1740 random (system-prompt, user-prompt) pairs from the 25 x 46 prompts of
Appendix G, generates two completions per pair -- one vanilla and one with
CFG at a guidance strength drawn from ``{1, ..., 6}`` -- and stores them so
that a (human or automatic) preference study can be run offline.

Example
-------
    python experiments/run_assistant_negative_prompting.py \
        --model "TheBloke/GPT4All-J-v1.3-jazzy" --n-pairs 1740 \
        --output results/assistant_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfglm.assistant_prompts import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    SYSTEM_PROMPT_SUFFIXES,
    USER_PROMPTS,
    build_prompt,
)
from cfglm.generation import cfg_generate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="TheBloke/GPT4All-J-v1.3-jazzy")
    parser.add_argument("--n-pairs", type=int, default=1740)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16" if torch.cuda.is_available() else "float32")
    parser.add_argument("--output", default="results/assistant_pairs.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pairs = [
        (rng.choice(SYSTEM_PROMPT_SUFFIXES), rng.choice(USER_PROMPTS))
        for _ in range(args.n_pairs)
    ]

    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype).to(args.device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        for i, (system_suffix, user_prompt) in enumerate(pairs):
            gamma = rng.choice([1, 2, 3, 4, 5, 6])
            prompt = build_prompt(system_suffix, user_prompt)
            vanilla = cfg_generate(
                model, tokenizer, prompt, gamma=1.0,
                max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7, seed=i,
            )[0]
            guided = cfg_generate(
                model, tokenizer, prompt, gamma=float(gamma),
                negative_prompt=DEFAULT_SYSTEM_PROMPT,
                max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7, seed=i,
            )[0]
            fh.write(
                json.dumps(
                    {
                        "system_prompt": system_suffix,
                        "user_prompt": user_prompt,
                        "gamma": gamma,
                        "vanilla": vanilla,
                        "cfg": guided,
                    }
                )
                + "\n"
            )
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(pairs)} pairs")
    print(f"Wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
