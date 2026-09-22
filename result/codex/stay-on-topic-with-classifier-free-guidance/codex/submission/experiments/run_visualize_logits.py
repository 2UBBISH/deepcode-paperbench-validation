#!/usr/bin/env python3
"""Section 5.3 / Table 3 -- visualising the vocabulary re-ordering by CFG.

Given a prompt, the script decodes a continuation and, at every step, ranks
the vocabulary by the guidance-induced change in log-probability

    delta(w) = gamma * ( log P(w | w_<t, c) - log P(w | w_<t) )

printing the most up-weighted and most down-weighted tokens in the layout of
the paper's Table 3.  The default prompt is the one used in the paper
(``The dragon flew over Paris, France``); the resulting ranking shows tokens
about dragons and Paris being encouraged while tokens about other locations
("Queensland"), dates ("1913") and topics ("hostages", "voyages") are
discouraged.

Example
-------
    python experiments/run_visualize_logits.py --model gpt2-large \
        --prompt "The dragon flew over Paris, France" --gamma 1.5 \
        --output results/vocab_ranking.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfglm.distributions import format_ranking_table, rank_tokens_by_guidance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gpt2-large")
    parser.add_argument("--prompt", default="The dragon flew over Paris, France")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--steps", type=int, default=16,
                        help="number of decoding steps to display")
    parser.add_argument("--uncond-prefix-tokens", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16" if torch.cuda.is_available() else "float32")
    parser.add_argument("--output", default="results/vocab_ranking.txt")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype).to(args.device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = rank_tokens_by_guidance(
        model,
        tokenizer,
        args.prompt,
        gamma=args.gamma,
        uncond_prefix_tokens=args.uncond_prefix_tokens,
        top_n=args.top_n,
        device=args.device,
    )[: args.steps]

    table = format_ranking_table(rows, top_n=args.top_n)
    print(f"\nPrompt: {args.prompt!r}   gamma={args.gamma}\n")
    print(table)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        fh.write(f"Prompt: {args.prompt!r}  gamma={args.gamma}\n\n")
        fh.write(table + "\n")
    if args.json_output:
        with open(args.json_output, "w") as fh:
            json.dump(rows, fh, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
