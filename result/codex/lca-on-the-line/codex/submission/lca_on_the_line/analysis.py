"""Turns the cached evaluation results into the paper's tables and figures.

* Table 1  -- model performance vs mistake severity
* Table 2  -- R^2 / PEA of ID LCA / Top-1 against OOD Top-1 / Top-5
* Table 3  -- MAE of the OOD error predictors (ID Top1, AC, Aline-S, Aline-D,
              ID LCA)
* Figure 1 -- LCA unifies VMs and VLMs (ObjectNet)
* Figure 5 -- the same story across the four severely shifted OOD datasets
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .baselines import evaluate_error_predictors
from .evaluate import OOD_DATASETS
from .metrics import pea, r2

DATASET_LABELS = {
    "imagenet": "ImageNet",
    "imagenet_v2": "ImgN-v2",
    "imagenet_s": "ImgN-S",
    "imagenet_r": "ImgN-R",
    "imagenet_a": "ImgN-A",
    "objectnet": "ObjNet",
}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_metrics(out_dir: str):
    import pandas as pd

    return pd.read_csv(os.path.join(out_dir, "metrics.csv"))


def metric_table(df, metric: str):
    """``index=model`` x ``columns=dataset`` for one metric."""
    return df.pivot_table(index="model", columns="dataset", values=metric)


def load_logits(out_dir: str, model: str, dataset: str) -> np.ndarray:
    safe = model.replace("/", "_").replace("@", "_").replace(" ", "")
    path = os.path.join(out_dir, "logits", "%s__%s.npy" % (safe, dataset))
    return np.load(path).astype(np.float64)


def load_targets(out_dir: str, dataset: str) -> np.ndarray:
    return np.load(os.path.join(out_dir, "targets", "%s.npy" % dataset))


def families(df) -> Dict[str, str]:
    return dict(zip(df["model"], df["family"]))


# --------------------------------------------------------------------------- #
# Table 1 / Table 2
# --------------------------------------------------------------------------- #
def compute_table1(df, datasets: Optional[Sequence[str]] = None):
    """Mistake severity (LCA) vs accuracy for the selected models."""
    import pandas as pd

    datasets = list(datasets or ["imagenet"] + list(OOD_DATASETS))
    lca = metric_table(df, "lca")
    top1 = metric_table(df, "top1")
    rows = []
    for model in lca.index:
        row = {"model": model}
        for ds in datasets:
            if ds in lca.columns:
                row["%s_lca" % DATASET_LABELS[ds]] = lca.loc[model, ds]
            if ds in top1.columns:
                row["%s_top1" % DATASET_LABELS[ds]] = top1.loc[model, ds]
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def compute_table2(df, ood_datasets: Sequence[str] = OOD_DATASETS) -> Dict:
    """R^2 / PEA of ``{ID Top1, ID LCA} x {OOD Top1, OOD Top5}``."""
    lca = metric_table(df, "lca")["imagenet"]
    top1 = metric_table(df, "top1")["imagenet"]
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for id_name, id_values in (("Top1", top1), ("LCA", lca)):
        for ood_metric in ("top1", "top5"):
            ood = metric_table(df, ood_metric)
            key = "%s->%s" % (id_name, ood_metric.upper())
            table[key] = {}
            for ds in ood_datasets:
                if ds not in ood.columns:
                    continue
                x = id_values.reindex(ood.index).values
                y = ood[ds].values
                table[key][DATASET_LABELS[ds]] = {
                    "R2": r2(x, y),
                    "PEA": pea(x, y),
                }
    return table


def format_table2(table: Dict) -> str:
    datasets = list(next(iter(table.values())).keys())
    header = "%-12s" % "Element" + "".join(
        "%-20s" % d for d in datasets
    )
    lines = [header]
    for key, values in table.items():
        cells = "".join(
            "%-20s" % ("%.3f / %.3f" % (values[d]["R2"], values[d]["PEA"]))
            for d in datasets
        )
        lines.append("%-12s%s" % (key, cells))
    return "\n".join(lines)


def compute_ranking_table(df, ood_datasets: Sequence[str] = OOD_DATASETS) -> Dict:
    """KEN / SPE of ``{ID Top1, ID LCA} x {OOD Top1, OOD Top5}``.

    The ranking measures (Kendall and Spearman) are introduced in the
    "Metric Setup" of Section 4 and reported per dataset in Appendix F.3; they
    complement the linearity measurements of Table 2.
    """
    from .metrics import ken, spearman

    lca = metric_table(df, "lca")["imagenet"]
    top1 = metric_table(df, "top1")["imagenet"]
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for id_name, id_values in (("Top1", top1), ("LCA", lca)):
        for ood_metric in ("top1", "top5"):
            ood = metric_table(df, ood_metric)
            key = "%s->%s" % (id_name, ood_metric.upper())
            table[key] = {}
            for ds in ood_datasets:
                if ds not in ood.columns:
                    continue
                x = id_values.reindex(ood.index).values
                y = ood[ds].values
                table[key][DATASET_LABELS[ds]] = {
                    "KEN": ken(x, y),
                    "SPE": spearman(x, y),
                }
    return table


def format_ranking_table(table: Dict) -> str:
    datasets = list(next(iter(table.values())).keys())
    lines = ["%-12s" % "Element" + "".join("%-20s" % d for d in datasets)]
    for key, values in table.items():
        cells = "".join(
            "%-20s" % ("%.3f / %.3f" % (values[d]["KEN"], values[d]["SPE"]))
            for d in datasets
        )
        lines.append("%-12s%s" % (key, cells))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Table 3
# --------------------------------------------------------------------------- #
def compute_table3(
    df,
    out_dir: str,
    ood_datasets: Sequence[str] = OOD_DATASETS,
    scaling: str = "minmax",
) -> "object":
    """MAE of each OOD error predictor (Table 3), overall.

    Logits are streamed one model at a time so that the full 75-model sweep
    never has to be held in memory at once.
    """
    import pandas as pd

    from .baselines import average_confidence
    from .lca import softmax

    models = list(df["model"].unique())
    lca_table = metric_table(df, "lca")
    top1_table = metric_table(df, "top1")
    id_lca = lca_table["imagenet"].reindex(models).values
    id_acc = top1_table["imagenet"].reindex(models).values

    id_preds: List[np.ndarray] = []
    for m in models:
        logits = load_logits(out_dir, m, "imagenet")
        id_preds.append(logits.argmax(axis=1))

    rows = {}
    for ds in ood_datasets:
        ood_acc = top1_table[ds].reindex(models).values
        ood_preds: List[np.ndarray] = []
        ood_confidence: List[float] = []
        for m in models:
            logits = load_logits(out_dir, m, ds)
            ood_preds.append(logits.argmax(axis=1))
            ood_confidence.append(
                average_confidence(softmax(logits))
            )
        rows[DATASET_LABELS[ds]] = evaluate_error_predictors(
            id_accuracy=id_acc,
            id_lca=id_lca,
            ood_accuracy=ood_acc,
            id_preds=id_preds,
            ood_preds=ood_preds,
            ood_confidence=ood_confidence,
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures 1 and 5
# --------------------------------------------------------------------------- #
def figure5(df, out_path: str, ood_datasets: Sequence[str] = OOD_DATASETS):
    """Scatter of OOD Top-1/Top-5 against ID Top-1 (red) and ID LCA (green)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lca = metric_table(df, "lca")["imagenet"]
    top1 = metric_table(df, "top1")["imagenet"]
    fam = families(df)
    models = list(lca.index)

    fig, axes = plt.subplots(2, len(ood_datasets), figsize=(4 * len(ood_datasets), 8))
    if len(ood_datasets) == 1:
        axes = axes.reshape(2, 1)
    for col, ds in enumerate(ood_datasets):
        ood1 = metric_table(df, "top1")[ds].reindex(models).values
        ood5 = metric_table(df, "top5")[ds].reindex(models).values
        for row, ood_values, ood_label in ((0, ood1, "Top1"), (1, ood5, "Top5")):
            ax = axes[row, col]
            ax2 = ax.twinx()
            for family, colour in (("VM", "red"), ("VLM", "blue")):
                idx = [i for i, m in enumerate(models) if fam[m] == family]
                ax.scatter(ood_values[idx], top1.reindex(models).values[idx],
                           s=14, c=colour, alpha=0.8, label="%s ID Top1" % family)
                ax2.scatter(ood_values[idx], lca.reindex(models).values[idx],
                            s=14, c="green", marker="x", alpha=0.8,
                            label="%s ID LCA" % family)
            ax.set_xlabel("OOD %s" % ood_label)
            ax.set_ylabel("ID Top-1", color="red")
            ax2.set_ylabel("ID LCA", color="green")
            ax.set_title("%s -- OOD %s" % (DATASET_LABELS[ds], ood_label))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def figure1(df, out_path: str, ood_dataset: str = "objectnet"):
    """The two-panel Figure 1 (accuracy-on-the-line vs LCA-on-the-line)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(metric_table(df, "lca").index)
    fam = families(df)
    ood = metric_table(df, "top1")[ood_dataset].reindex(models).values
    id_top1 = metric_table(df, "top1")["imagenet"].reindex(models).values
    id_lca = metric_table(df, "lca")["imagenet"].reindex(models).values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for family, colour in (("VM", "red"), ("VLM", "blue")):
        idx = [i for i, m in enumerate(models) if fam[m] == family]
        axes[0].scatter(ood[idx], id_top1[idx], s=16, c=colour, label=family)
        axes[1].scatter(ood[idx], -id_lca[idx], s=16, c="green" if family == "VM" else "steelblue",
                        label="%s" % family)
    axes[0].set_xlabel("ObjectNet top-1")
    axes[0].set_ylabel("ImageNet top-1 (ID)")
    axes[0].set_title("Accuracy-on-the-line")
    axes[1].set_xlabel("ObjectNet top-1")
    axes[1].set_ylabel("- ImageNet LCA (ID)")
    axes[1].set_title("LCA-on-the-line")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def figure9(df, out_path: str,
            datasets: Sequence[str] = ("imagenet", "imagenet_v2") + OOD_DATASETS):
    """LCA vs Top-1 *on the same dataset* (Appendix F.1, Figure 9)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(metric_table(df, "lca").index)
    fam = families(df)
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4))
    for ax, ds in zip(np.atleast_1d(axes), datasets):
        top1 = metric_table(df, "top1")[ds].reindex(models).values
        lca = metric_table(df, "lca")[ds].reindex(models).values
        for family, colour in (("VM", "red"), ("VLM", "blue")):
            idx = [i for i, m in enumerate(models) if fam[m] == family]
            ax.scatter(top1[idx], lca[idx], s=16, c=colour, label=family)
        ax.set_xlabel("%s Top-1" % DATASET_LABELS[ds])
        ax.set_ylabel("%s LCA" % DATASET_LABELS[ds])
    np.atleast_1d(axes)[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Figure 7 / Figure 8 (pairwise LCA matrices and soft-label quality)
# --------------------------------------------------------------------------- #
def figure7(matrices: Dict[str, np.ndarray], out_path: str):
    """Pairwise LCA distance matrices, rows sorted ascending (Figure 7).

    ``matrices`` maps a hierarchy name (``"WordNet"``, ``"ResNet50"``,
    ``"CLIP_RN50"``, ...) to its ``(1000, 1000)`` LCA distance matrix.  Each row
    is sorted ascending, so the diagonal index becomes the shortest distance.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(matrices)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        matrix = np.asarray(matrices[name], dtype=np.float64)
        sorted_rows = np.sort(matrix, axis=1)
        im = ax.imshow(sorted_rows, aspect="auto", cmap="viridis")
        ax.set_title(name)
        ax.set_xlabel("class pairs (sorted)")
        ax.set_ylabel("class")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def figure8(source_lca: np.ndarray, probe_ood_accuracy: np.ndarray,
            out_path: str, dataset_label: str = "ImageNet-A"):
    """Source-model ID LCA vs the OOD accuracy of the probe it trained."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    ax.scatter(source_lca, probe_ood_accuracy, s=16, c="tab:blue")
    ax.set_xlabel("source model ID LCA (WordNet)")
    ax.set_ylabel("linear-probe OOD Top-1 (%s)" % dataset_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
