"""Figures and tables of the paper, rendered from the JSON results.

* **Figure 1**: radar plot comparing CLIP / TeCoA / FARE across the zero-shot
  and LLaVA tasks.  Per App. B.2 each radial axis runs from 0 to the maximum
  value across the compared models, and the maximum is printed next to the axis
  label.  "ZS-Class." is the average zero-shot classification accuracy of the
  datasets of Sec. 4.3.
* **Tables 1/3/4**: compact LaTeX/markdown renderings of the collected metrics,
  which makes it easy to check the trends against the paper.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional


#: friendly axis labels -> the key inside one of the result jsons
FIGURE1_AXES = {
    "ZS-Class.": "average.avg_zero_shot_clean",
    "COCO": "clean_cider_coco",
    "Flickr30k": "clean_cider_flickr30k",
    "TextVQA": "clean_accuracy_textvqa",
    "VQAv2": "clean_accuracy_vqav2",
}


def _dig(data: dict, path: str):
    current = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def collect_figure1(results: Dict[str, List[dict]]) -> Dict[str, Dict[str, float]]:
    """``{model: {axis: value}}`` from the per-task result files."""
    collected: Dict[str, Dict[str, float]] = {}
    for model, files in results.items():
        per_axis: Dict[str, float] = {}
        for axis, key in FIGURE1_AXES.items():
            for data in files:
                value = _dig(data, key)
                if value is not None:
                    per_axis[axis] = float(value)
                    break
        collected[model] = per_axis
    return collected


def radar_plot(per_model: Dict[str, Dict[str, float]], output: str, title: str = "") -> str:
    """The radar plot of Figure 1."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    axes = [a for a in FIGURE1_AXES if all(a in v for v in per_model.values())]
    if not axes:
        raise ValueError("no common axes across the models; check the input files")
    maxima = {a: max(v[a] for v in per_model.values()) for a in axes}

    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
    angles += angles[:1]

    figure, axis = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
    for (model, values), color in zip(per_model.items(), colors):
        radii = [values[a] / maxima[a] for a in axes]
        radii += radii[:1]
        axis.plot(angles, radii, label=model, color=color, linewidth=2)
        axis.fill(angles, radii, color=color, alpha=0.08)
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels([f"{a}\n(max {maxima[a]:.1f})" for a in axes], fontsize=9)
    axis.set_yticklabels([])
    axis.set_title(title or "Robust CLIP (Fig. 1)", pad=20)
    axis.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    return output


def latex_table_1(per_model: Dict[str, Dict[str, float]]) -> str:
    """Render the captioning / VQA part of Table 1."""
    axes = ["COCO", "Flickr30k", "TextVQA", "VQAv2"]
    rows = ["Model & " + " & ".join(axes) + " \\\\", "\\hline"]
    for model, values in per_model.items():
        rows.append(
            "{} & {} \\\\".format(model, " & ".join(_fmt(values.get(a)) for a in axes))
        )
    return "\n".join(rows)


def _fmt(value: Optional[float]) -> str:
    return "--" if value is None else f"{value:.1f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results", nargs="+", required=True,
        help="model=path.json entries (one file per task, comma separated): "
             "e.g. CLIP=zs.json,coco.json,vqa.json",
    )
    parser.add_argument("--out", default="results/figure1.png")
    parser.add_argument("--title", default="")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    results: Dict[str, List[dict]] = {}
    for entry in args.results:
        model, paths = entry.split("=", 1)
        files = []
        for path in paths.split(","):
            with open(path, "r", encoding="utf-8") as handle:
                files.append(json.load(handle))
        results[model] = files
    per_model = collect_figure1(results)
    path = radar_plot(per_model, args.out, args.title)
    print(f"wrote {path}")
    print(latex_table_1(per_model))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
