#!/usr/bin/env python
"""Draw the figure analogues of the paper (Figures 1-5) from saved artifacts.

Each figure is optional: the script skips artifacts that have not been produced
yet, so it can be run incrementally while the pipeline progresses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402


def figure_logit_lens(path: Path, out: Path) -> None:
    from dpo_toxic.utils import load_json

    data = load_json(path)
    layers = range(len(data["gpt2"]["post_block"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(layers), data["gpt2"]["post_block"], marker="o", label="GPT2")
    ax.plot(list(layers), data["gpt2_dpo"]["post_block"], marker="s", label="GPT2_DPO")
    ax.set_xlabel("layer")
    ax.set_ylabel(f"P({data['gpt2'].get('token', 'sh*t')})")
    ax.set_title(f"Logit lens ({data.get('n_prompts', '?')} prompts)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figure1_logit_lens.png", dpi=150)
    plt.close(fig)


def figure_activation_drop(path: Path, out: Path) -> None:
    from dpo_toxic.utils import load_json

    data = load_json(path)
    rows = data["rows"]
    labels = [f"v_{r['index']}^{r['layer']}" for r in rows]
    before = [r["mean_activation_gpt2"] for r in rows]
    after = [r["mean_activation_dpo"] for r in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - 0.2 for i in x], before, width=0.4, label="GPT2")
    ax.bar([i + 0.2 for i in x], after, width=0.4, label="GPT2_DPO")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("mean activation $m_i$")
    ax.set_title("Toxic-vector activations before/after DPO (Fig. 2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figure2_activation_drop.png", dpi=150)
    plt.close(fig)


def figure_pca(layer: int, artifacts: Path, out: Path) -> None:
    blob = torch.load(artifacts / f"pca_projection_layer{layer}.pt", map_location="cpu")
    fig, ax = plt.subplots(figsize=(6, 5))
    act_before = blob.get("activates_gpt2")
    act_after = blob.get("activates_gpt2_dpo")
    if act_before is not None and act_after is not None:
        for act, xs, ys, marker, label in (
                (act_before, blob["delta_axis_before"], blob["pc_axis_before"], "o", "GPT2"),
                (act_after, blob["delta_axis_after"], blob["pc_axis_after"], "^", "GPT2_DPO")):
            active = act.bool()
            ax.scatter(xs[active], ys[active], s=10, marker=marker, color="tab:red",
                       alpha=0.7, label=f"{label} (activates)")
            ax.scatter(xs[~active], ys[~active], s=10, marker=marker, color="tab:blue",
                       alpha=0.7, label=f"{label} (inactive)")
    else:
        ax.scatter(blob["delta_axis_before"], blob["pc_axis_before"], s=6, alpha=0.5,
                   label="GPT2", marker="o")
        ax.scatter(blob["delta_axis_after"], blob["pc_axis_after"], s=6, alpha=0.5,
                   label="GPT2_DPO", marker="^")
    ax.set_xlabel(r"$\bar{\delta}_x$")
    ax.set_ylabel("principal component")
    ax.set_title(f"Residual streams at layer {layer} (Fig. 4)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"figure4_pca_layer{layer}.png", dpi=150)
    plt.close(fig)


def figure_shift_vs_delta(layer: int, artifacts: Path, out: Path) -> None:
    cosines = torch.load(artifacts / f"delta_cosine_layer{layer}.pt", map_location="cpu")
    acts = torch.load(artifacts / f"value_vector_activations_layer{layer}.pt", map_location="cpu")
    layers = sorted(cosines.keys())
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for l in layers:
        c = cosines[l].numpy()
        axes[0].hist(c, bins=40, alpha=0.4, label=f"layer {l}" if l % 4 == 0 else None)
    axes[0].set_ylabel("% of value vectors")
    axes[0].set_title(f"cos($\\delta_x^{{{layer}}}$, $\\delta_{{MLP.v}}$) per layer (Fig. 5)")
    axes[0].legend()
    axes[1].hist(acts.numpy(), bins=60, color="tab:orange", alpha=0.7)
    axes[1].set_xlabel("cosine similarity / mean activation")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"mean activations of value vectors at layer {layer}")
    fig.tight_layout()
    fig.savefig(out / f"figure5_delta_cosine_layer{layer}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--layer", type=int, default=19)
    args = parser.parse_args()
    artifacts = Path(args.artifacts)
    figures = artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    made = []
    candidates = [
        (artifacts / "logit_lens", figure_logit_lens, "figure1_logit_lens.png"),
        (artifacts / "analysis" / "activation_drop.json", figure_activation_drop,
         "figure2_activation_drop.png"),
    ]
    for path, fn, name in candidates:
        if path.exists():
            target = path if path.is_file() else sorted(path.glob("*.json"))[0]
            fn(target, figures)
            made.append(name)
    if (artifacts / "analysis" / f"pca_projection_layer{args.layer}.pt").exists():
        figure_pca(args.layer, artifacts / "analysis", figures)
        made.append(f"figure4_pca_layer{args.layer}.png")
    if (artifacts / "analysis" / f"delta_cosine_layer{args.layer}.pt").exists():
        figure_shift_vs_delta(args.layer, artifacts / "analysis", figures)
        made.append(f"figure5_delta_cosine_layer{args.layer}.png")
    print("figures written:", made)


if __name__ == "__main__":
    main()
