#!/usr/bin/env python
"""Section 3.2 -- project W_toxic, the toxic value vectors and SVD.U_toxic onto
the vocabulary space (Table 1 / Table 6).
"""

from __future__ import annotations

import argparse

from _common import add_common_args, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--vectors-path", default="artifacts/toxic_vectors/toxic_vectors.pt")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--n-svd", type=int, default=5)
    args = parser.parse_args()

    from dpo_toxic.toxic_vectors import load_toxic_vectors
    from dpo_toxic.utils import load_json
    from dpo_toxic.vocab_projection import project_and_save

    model, tokenizer = load_model_and_tokenizer(args)
    blob = load_toxic_vectors(args.vectors_path)
    meta_path = str(args.vectors_path).replace("toxic_vectors.pt", "toxic_vectors.json")
    selections = load_json(meta_path)["selections"]
    table = project_and_save(model, tokenizer, blob["w_toxic"], selections,
                             blob["raw_value_vectors"], blob["svd_u"],
                             out_dir=f"{args.out_dir}/vocab", k=args.k, n_svd=args.n_svd)
    for name, rows in table.items():
        print(f"{name}: " + ", ".join(r["token"] for r in rows))


if __name__ == "__main__":
    main()
