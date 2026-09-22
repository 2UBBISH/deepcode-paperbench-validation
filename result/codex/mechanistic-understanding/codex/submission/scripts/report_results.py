#!/usr/bin/env python
"""Assemble a human-readable summary of whatever artifacts exist.

    python scripts/report_results.py --artifacts artifacts

writes ``artifacts/RESULTS.md`` with the probe accuracy, the ranking statistics
of the toxic value vectors, the vocabulary projection (Table 1 analogue) and any
evaluation tables that have been produced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    art = Path(args.artifacts)
    lines = ["# Reproduction results", ""]

    probe = _load(art / "probe" / "probe_report.json")
    if probe:
        lines += [
            "## Section 3.1 -- toxicity probe",
            "",
            f"* model: `{probe['model_name']}`, layer {probe['config']['layer']}, "
            f"max_length {probe['config']['max_length']}",
            f"* data: {probe['data']['n_train']} train / {probe['data']['n_val']} validation "
            f"comments (90:10 split)",
            f"* validation accuracy: **{probe['metrics']['best_val_acc']:.4f}** "
            f"(paper: 0.94)",
            "",
        ]

    vec_meta = _load(art / "toxic_vectors" / "toxic_vectors.json")
    if vec_meta:
        rows = vec_meta["selections"][:10]
        lines += [
            "## Section 3.1 -- toxic value vectors (top-10 of "
            f"{vec_meta['n_value_vectors']} selected)",
            "",
            "| rank | vector | cosine with W_toxic |",
            "|---|---|---|",
        ]
        for i, r in enumerate(rows):
            lines.append(f"| {i} | MLP.v_{r['index']}^{r['layer']} | {r['cosine']:.3f} |")
        evr = vec_meta.get("explained_variance_ratio", [])
        lines += ["", "explained variance of the first SVD directions: "
                  + ", ".join(f"{v:.3f}" for v in evr[:5]), ""]
    vocab = _load(art / "vocab" / "vocab_projection.json")
    if vocab:
        lines += ["## Section 3.2 -- vocabulary projection (Table 1 analogue)", "",
                  "| Vector | top tokens |", "|---|---|"]
        for name, row in vocab.items():
            lines.append(f"| {name} | " + ", ".join(r["token"] for r in row) + " |")
        lines.append("")

    interventions = _load(art / "interventions" / "intervention_table.json")
    if interventions:
        lines += ["## Section 3.3 -- interventions (Table 2 analogue)", "",
                  "| Method | Toxicity | PPL | F1 | alpha |", "|---|---|---|---|---|"]
        for name, row in interventions.items():
            lines.append("| {} | {} | {} | {} | {} |".format(
                name,
                _fmt(row.get("toxicity")), _fmt(row.get("ppl")), _fmt(row.get("f1")),
                _fmt(row.get("alpha"))))
        lines.append("")

    unalign = _load(art / "unalign" / "table4.json")
    if unalign:
        lines += ["## Section 6 -- un-aligning GPT2_DPO (Table 4 analogue)", "",
                  "| Method | Toxicity | PPL | F1 |", "|---|---|---|---|"]
        for name, row in unalign.items():
            lines.append("| {} | {} | {} | {} |".format(
                name, _fmt(row.get("toxicity")), _fmt(row.get("ppl")), _fmt(row.get("f1"))))
        lines.append("")

    shift = _load(art / "analysis" / "parameter_shift.json")
    if shift:
        s = shift["summary"]
        lines += ["## Section 5.1 -- parameter shift",
                  "",
                  f"* tensors compared: {s['n_tensors']}",
                  f"* minimum cosine similarity: {s['min_cosine']:.6f} "
                  f"(tensors below 0.99: {s['n_below_0_99']})",
                  f"* mean |difference|: {s['mean_abs_diff']:.3e}",
                  ""]

    acts = _load(art / "analysis" / "activation_drop.json")
    if acts:
        lines += ["## Section 5.2 -- toxic-vector activations before/after DPO (Fig. 2)", "",
                  "| vector | GPT2 | GPT2_DPO | drop |", "|---|---|---|---|"]
        for r in acts["rows"]:
            lines.append(f"| MLP.v_{r['index']}^{r['layer']} | {r['mean_activation_gpt2']:.4f} "
                         f"| {r['mean_activation_dpo']:.4f} | {r['drop']:.4f} |")
        lines.append("")

    eval_dir = art
    evals = sorted(eval_dir.glob("eval_*.json"))
    if evals:
        lines += ["## Toxicity / PPL / F1 evaluations", "",
                  "| model | toxicity | PPL | F1 |", "|---|---|---|---|"]
        for f in evals:
            d = _load(f)
            lines.append(f"| {d.get('tag', f.stem)} | {_fmt(d.get('toxicity'))} "
                         f"| {_fmt(d.get('ppl'))} | {_fmt(d.get('f1_f1') or d.get('f1'))} |")
        lines.append("")

    out = art / "RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    main()
