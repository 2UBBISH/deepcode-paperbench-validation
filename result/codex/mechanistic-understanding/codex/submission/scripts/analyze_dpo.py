#!/usr/bin/env python
"""Section 5 -- the post-DPO analyses.

Runs, for a (pre-DPO, post-DPO) model pair:

* Section 5.1: parameter shifts (cosine similarity / norm difference)
* Section 5.2 / Figure 2: mean activations of the toxic vectors
* Section 5.2 / Figures 3-4: the residual-stream offset ``delta_x`` and the PCA view
* Section 5.2 / Figure 5: ``cos(delta_x, delta_MLP.v)`` and value-vector activations
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import add_common_args, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--dpo-model", default="artifacts/dpo/model")
    parser.add_argument("--vectors-path", default="artifacts/toxic_vectors/toxic_vectors.pt")
    parser.add_argument("--layer", type=int, default=19,
                        help="layer used for the residual-stream figures (paper: 19)")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--generated-tokens", type=int, default=20)
    parser.add_argument("--top-vectors", type=int, default=5)
    args = parser.parse_args()

    import torch

    from dpo_toxic.analysis.activations import activation_drop_table
    from dpo_toxic.analysis.parameter_shift import compare_parameters, summarize_comparison
    from dpo_toxic.analysis.residual_shift import (
        mean_residual_shift, mean_value_vector_activations, pca_projection,
        per_prompt_activations, shift_vs_value_vector_shift)
    from dpo_toxic.data.realtoxicity import load_realtoxicity_challenge
    from dpo_toxic.utils import load_json, save_json, set_seed

    set_seed(args.seed)
    before_model, tokenizer = load_model_and_tokenizer(args)
    after_model, _ = load_model_and_tokenizer(args, path=args.dpo_model)

    meta = load_json(str(args.vectors_path).replace(".pt", ".json"))
    selections = meta["selections"][: args.top_vectors]
    out = Path(args.out_dir) / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 5.1
    comparison = compare_parameters(before_model, after_model)
    summary = summarize_comparison(comparison)
    print("parameter shift:", summary)
    save_json({"comparison": comparison, "summary": summary}, out / "parameter_shift.json")

    prompts = load_realtoxicity_challenge(cache_dir=args.cache_dir)
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]

    # ---------------------------------------------------------------- Figure 2
    act_table = activation_drop_table(before_model, after_model, tokenizer, prompts,
                                      selections, batch_size=args.batch_size,
                                      device=args.device,
                                      max_new_tokens=args.generated_tokens)
    save_json(act_table, out / "activation_drop.json")
    for row in act_table["rows"]:
        print(f"MLP.v_{row['index']}^{row['layer']}: "
              f"GPT2={row['mean_activation_gpt2']:.4f} DPO={row['mean_activation_dpo']:.4f}")

    # ---------------------------------------------------------------- Figures 3-4
    shift = mean_residual_shift(before_model, after_model, tokenizer, prompts,
                                layer=args.layer, batch_size=args.batch_size, device=args.device)
    torch.save({"x_before": shift["x_before"], "x_after": shift["x_after"],
                "delta": shift["delta"], "mean_delta": shift["mean_delta"]},
               out / f"residual_shift_layer{args.layer}.pt")

    proj = pca_projection(shift["x_before"], shift["x_after"], shift["mean_delta"])
    # colouring of Figure 4: does each residual stream activate the top toxic
    # value vector at this layer?
    top = selections[0]
    activates = {
        "gpt2": per_prompt_activations(before_model, tokenizer, prompts, layer=args.layer,
                                       index=int(top["index"]), batch_size=args.batch_size,
                                       device=args.device),
        "gpt2_dpo": per_prompt_activations(after_model, tokenizer, prompts, layer=args.layer,
                                           index=int(top["index"]), batch_size=args.batch_size,
                                           device=args.device),
    }
    proj["activates_gpt2"] = activates["gpt2"]["activates"]
    proj["activates_gpt2_dpo"] = activates["gpt2_dpo"]["activates"]
    proj["activation_mean_gpt2"] = activates["gpt2"]["mean"]
    proj["activation_mean_gpt2_dpo"] = activates["gpt2_dpo"]["mean"]
    torch.save(proj, out / f"pca_projection_layer{args.layer}.pt")

    # ---------------------------------------------------------------- Figure 5
    cosines = shift_vs_value_vector_shift(before_model, after_model, shift["mean_delta"],
                                          layer=args.layer)
    torch.save(cosines, out / f"delta_cosine_layer{args.layer}.pt")
    act_vectors = mean_value_vector_activations(before_model, tokenizer, prompts,
                                                layer=args.layer,
                                                batch_size=args.batch_size, device=args.device)
    torch.save(act_vectors, out / f"value_vector_activations_layer{args.layer}.pt")
    n_neg = sum(int((c < 0).sum()) for c in cosines.values())
    n_tot = sum(int(c.numel()) for c in cosines.values())
    print(f"cos(delta_x^{args.layer}-mid, delta_MLP.v): {n_neg}/{n_tot} negative")


if __name__ == "__main__":
    main()
