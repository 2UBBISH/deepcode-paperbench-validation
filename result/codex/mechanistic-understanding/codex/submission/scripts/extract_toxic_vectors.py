#!/usr/bin/env python
"""Section 3.1 -- rank MLP value vectors by cosine similarity with W_toxic,
keep the top N = 128 and compute the SVD basis ``SVD.U_toxic``.
"""

from __future__ import annotations

import argparse

from _common import add_common_args, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--w-toxic", default="artifacts/probe/w_toxic.pt")
    parser.add_argument("--top-n", type=int, default=128)
    parser.add_argument("--n-svd", type=int, default=10)
    parser.add_argument("--branch", default="k", choices=["k", "up"])
    args = parser.parse_args()

    from dpo_toxic.toxic_vectors import ToxicVectorConfig, extract_toxic_vectors

    model, _ = load_model_and_tokenizer(args)
    cfg = ToxicVectorConfig(top_n=args.top_n, branch=args.branch, n_svd_components=args.n_svd)
    res = extract_toxic_vectors(model, args.w_toxic, cfg=cfg,
                                out_dir=f"{args.out_dir}/toxic_vectors")
    print("top-5 toxic value vectors:")
    for s in res["selection"]["selections"][:5]:
        print(f"  MLP.v_{s['index']}^{s['layer']}  cosine={s['cosine']:.3f}")


if __name__ == "__main__":
    main()
