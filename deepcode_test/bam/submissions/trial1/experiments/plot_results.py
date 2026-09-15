"""Plotting utilities for the Batch-and-Match (BaM) reproduction suite.

This module reads the tidy CSV files produced by the experiment runners and
reproduces the main qualitative figures from the paper:

* ``results/gaussian_targets.csv``   -> Section 5.1 / Figures 5.2 and E.2-E.3
* ``results/sinh_arcsinh.csv``       -> Section 5.1 non-Gaussian targets
* ``results/posteriordb.csv``        -> Section 5.2 / Figure 5.3
* ``results/cifar.csv``              -> Section 5.3 / Figure 5.4 and E.7
* ``results/*.csv``                  -> Appendix E.1 wallclock timings

The plotting functions are intentionally tolerant of missing columns and
missing files so that individual figures can be generated as soon as their
corresponding experiment has been run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUT_DIR = PROJECT_ROOT / "figures"

ALGORITHM_STYLE: Dict[str, Dict[str, object]] = {
    "bam": {"color": "#1f77b4", "marker": "o", "label": "BaM"},
    "advi": {"color": "#ff7f0e", "marker": "s", "label": "ADVI"},
    "score": {"color": "#2ca02c", "marker": "^", "label": "Score"},
    "fisher": {"color": "#d62728", "marker": "v", "label": "Fisher"},
    "gsm": {"color": "#9467bd", "marker": "D", "label": "GSM"},
    "avi": {"color": "#8c564b", "marker": "P", "label": "AVI"},
}


def _alg_style(algorithm: str) -> Dict[str, object]:
    """Return plotting style for an algorithm, falling back to a default."""
    base = ALGORITHM_STYLE.get(str(algorithm).lower(), {})
    return {
        "color": base.get("color", "#333333"),
        "marker": base.get("marker", "o"),
        "label": base.get("label", str(algorithm)),
    }


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV if it exists, otherwise return ``None``."""
    path = Path(path)
    if not path.exists():
        print(f"[plot_results] skipping missing file: {path}")
        return None
    df = pd.read_csv(path)
    print(f"[plot_results] loaded {path} with {len(df)} rows")
    return df


def _get_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first column name present in ``df``."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _savefig(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_results] wrote {path}")
    return path


def _mean_sem(
    df: pd.DataFrame, x_col: str, y_col: str, group_col: str = "algorithm"
) -> pd.DataFrame:
    """Aggregate Monte-Carlo runs into mean and standard-error columns."""
    grouped = (
        df.groupby([x_col, group_col], as_index=False)[y_col]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    # ``groupby(...).agg`` with as_index=False gives a column MultiIndex in some
    # pandas versions; flatten it defensively.
    if isinstance(grouped.columns, pd.MultiIndex):
        grouped.columns = [
            "_".join(str(c) for c in col).strip("_") for col in grouped.columns
        ]
    grouped.rename(
        columns={
            y_col + "_mean": "mean",
            y_col + "_sem": "sem",
            y_col + "_count": "count",
        },
        inplace=True,
    )
    return grouped


def _plot_alg_curves(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    log_y: bool = False,
    legend: bool = True,
) -> None:
    """Plot mean +/- sem curves for each algorithm over ``x_col``."""
    grouped = _mean_sem(df, x_col, y_col)
    for alg, sub in grouped.groupby("algorithm"):
        style = _alg_style(str(alg))
        ax.errorbar(
            sub[x_col],
            sub["mean"],
            yerr=sub["sem"],
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            capsize=3,
            linewidth=1.8,
            markersize=5,
        )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if log_y:
        ax.set_yscale("log")
    if legend:
        ax.legend(fontsize="small", frameon=False)
    ax.grid(alpha=0.25)


# ---------------------------------------------------------------------------
# Section 5.1: Gaussian targets
# ---------------------------------------------------------------------------

def plot_gaussian_targets(
    csv_path: Path,
    out_dir: Path,
    figures: Optional[Iterable[str]] = None,
) -> List[Path]:
    """Plot forward/reverse KL, gradient evaluations, and wallclock time.

    Parameters
    ----------
    csv_path:
        Path to ``results/gaussian_targets.csv``.
    out_dir:
        Directory in which to write PNG figures.
    figures:
        Optional subset of ``{"forward_kl", "reverse_kl", "grad_evals",
        "wallclock"}``.

    Returns
    -------
    List of written figure paths.
    """
    df = _read_csv(csv_path)
    if df is None:
        return []

    if figures is None:
        figures = {"forward_kl", "reverse_kl", "grad_evals", "wallclock"}
    figures = set(figures)

    dim_col = _get_col(df, ["dimension", "dim", "D"])
    alg_col = _get_col(df, ["algorithm", "algo", "method"])
    if dim_col is None or alg_col is None:
        print("[plot_results] Gaussian CSV missing dimension/algorithm columns")
        return []

    written: List[Path] = []

    fkl_col = _get_col(df, ["final_forward_kl", "forward_kl", "fkl"])
    rkl_col = _get_col(df, ["final_reverse_kl", "reverse_kl", "rkl"])
    grad_col = _get_col(df, ["grad_evals_to_threshold", "grad_evals", "n_grad"])
    time_col = _get_col(df, ["wallclock_seconds", "wallclock", "time"])

    if "forward_kl" in figures and fkl_col is not None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        _plot_alg_curves(ax, df, dim_col, fkl_col, log_y=True)
        ax.set_title("Gaussian targets: forward KL")
        written.append(_savefig(fig, out_dir, "gaussian_forward_kl.png"))

    if "reverse_kl" in figures and rkl_col is not None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        _plot_alg_curves(ax, df, dim_col, rkl_col, log_y=True)
        ax.set_title("Gaussian targets: reverse KL")
        written.append(_savefig(fig, out_dir, "gaussian_reverse_kl.png"))

    if "grad_evals" in figures and grad_col is not None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        _plot_alg_curves(ax, df, dim_col, grad_col, log_y=True)
        ax.set_title("Gaussian targets: gradient evaluations to threshold")
        written.append(_savefig(fig, out_dir, "gaussian_grad_evals.png"))

    if "wallclock" in figures and time_col is not None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        _plot_alg_curves(ax, df, dim_col, time_col, log_y=True)
        ax.set_title("Gaussian targets: wallclock time")
        written.append(_savefig(fig, out_dir, "gaussian_wallclock.png"))

    return written


# ---------------------------------------------------------------------------
# Section 5.1: sinh-arcsinh non-Gaussian targets
# ---------------------------------------------------------------------------

def plot_sinh_arcsinh(
    csv_path: Path,
    out_dir: Path,
    figures: Optional[Iterable[str]] = None,
) -> List[Path]:
    """Plot skew and tail-weight comparisons for sinh-arcsinh targets."""
    df = _read_csv(csv_path)
    if df is None:
        return []

    if figures is None:
        figures = {"forward_kl", "reverse_kl", "grad_evals"}
    figures = set(figures)

    case_col = _get_col(df, ["case_type", "case", "kind"])
    value_col = _get_col(df, ["case_value", "value", "skew", "tailweight"])
    alg_col = _get_col(df, ["algorithm", "algo", "method"])
    if case_col is None or value_col is None or alg_col is None:
        print("[plot_results] sinh-arcsinh CSV missing required columns")
        return []

    fkl_col = _get_col(df, ["final_forward_kl", "forward_kl", "fkl"])
    rkl_col = _get_col(df, ["final_reverse_kl", "reverse_kl", "rkl"])
    grad_col = _get_col(df, ["grad_evals_to_threshold", "grad_evals", "n_grad"])

    written: List[Path] = []

    for case_name in sorted(df[case_col].dropna().unique()):
        sub = df[df[case_col] == case_name]

        if "forward_kl" in figures and fkl_col is not None:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            _plot_alg_curves(ax, sub, value_col, fkl_col, log_y=True)
            ax.set_title(f"sinh-arcsinh ({case_name}): forward KL")
            written.append(_savefig(fig, out_dir, f"sinh_{case_name}_forward_kl.png"))

        if "reverse_kl" in figures and rkl_col is not None:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            _plot_alg_curves(ax, sub, value_col, rkl_col, log_y=True)
            ax.set_title(f"sinh-arcsinh ({case_name}): reverse KL")
            written.append(_savefig(fig, out_dir, f"sinh_{case_name}_reverse_kl.png"))

        if "grad_evals" in figures and grad_col is not None:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            _plot_alg_curves(ax, sub, value_col, grad_col, log_y=True)
            ax.set_title(f"sinh-arcsinh ({case_name}): gradient evaluations")
            written.append(_savefig(fig, out_dir, f"sinh_{case_name}_grad_evals.png"))

    return written


# ---------------------------------------------------------------------------
# Section 5.2: hierarchical posterior targets (posteriordb)
# ---------------------------------------------------------------------------

def plot_posteriordb(
    csv_path: Path,
    out_dir: Path,
    figures: Optional[Iterable[str]] = None,
) -> List[Path]:
    """Plot relative posterior mean/SD errors for posteriordb models."""
    df = _read_csv(csv_path)
    if df is None:
        return []

    if figures is None:
        figures = {"mean_error", "sd_error"}
    figures = set(figures)

    model_col = _get_col(df, ["model", "posterior", "target"])
    alg_col = _get_col(df, ["algorithm", "algo", "method"])
    if model_col is None or alg_col is None:
        print("[plot_results] posteriordb CSV missing model/algorithm columns")
        return []

    mean_col = _get_col(df, ["mean_rel_error", "relative_mean_error", "mean_error"])
    sd_col = _get_col(df, ["sd_rel_error", "relative_sd_error", "sd_error"])

    written: List[Path] = []

    if "mean_error" in figures and mean_col is not None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        grouped = _mean_sem(df, model_col, mean_col)
        models = sorted(grouped[model_col].unique())
        algs = sorted(grouped["algorithm"].unique())
        x = np.arange(len(models))
        width = 0.8 / max(len(algs), 1)
        for i, alg in enumerate(algs):
            sub = grouped[grouped["algorithm"] == alg].set_index(model_col).reindex(models)
            style = _alg_style(str(alg))
            ax.bar(
                x + (i - (len(algs) - 1) / 2) * width,
                sub["mean"].values,
                width,
                yerr=sub["sem"].fillna(0).values,
                label=style["label"],
                color=style["color"],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel("relative posterior mean error")
        ax.set_yscale("log")
        ax.legend(fontsize="small", frameon=False)
        ax.grid(alpha=0.25, axis="y")
        written.append(_savefig(fig, out_dir, "posteriordb_mean_error.png"))

    if "sd_error" in figures and sd_col is not None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        grouped = _mean_sem(df, model_col, sd_col)
        models = sorted(grouped[model_col].unique())
        algs = sorted(grouped["algorithm"].unique())
        x = np.arange(len(models))
        width = 0.8 / max(len(algs), 1)
        for i, alg in enumerate(algs):
            sub = grouped[grouped["algorithm"] == alg].set_index(model_col).reindex(models)
            style = _alg_style(str(alg))
            ax.bar(
                x + (i - (len(algs) - 1) / 2) * width,
                sub["mean"].values,
                width,
                yerr=sub["sem"].fillna(0).values,
                label=style["label"],
                color=style["color"],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel("relative posterior SD error")
        ax.set_yscale("log")
        ax.legend(fontsize="small", frameon=False)
        ax.grid(alpha=0.25, axis="y")
        written.append(_savefig(fig, out_dir, "posteriordb_sd_error.png"))

    return written


# ---------------------------------------------------------------------------
# Section 5.3: CIFAR-10 deep generative model
# ---------------------------------------------------------------------------

def _plot_cifar_mse(csv_path: Path, out_dir: Path) -> List[Path]:
    """Plot reconstruction MSE for CIFAR-10 experiments."""
    df = _read_csv(csv_path)
    if df is None:
        return []

    alg_col = _get_col(df, ["algorithm", "algo", "method"])
    mse_col = _get_col(df, ["recon_mse", "mse", "reconstruction_mse"])
    batch_col = _get_col(df, ["B", "batch_size", "batch"])

    if alg_col is None or mse_col is None:
        print("[plot_results] CIFAR CSV missing algorithm/mse columns")
        return []

    # Aggregate over runs.
    grouped = df.groupby([alg_col] + ([batch_col] if batch_col else []), as_index=False)[
        mse_col
    ].agg(["mean", "sem"])
    if isinstance(grouped.columns, pd.MultiIndex):
        grouped.columns = [
            "_".join(str(c) for c in col).strip("_") for col in grouped.columns
        ]
    grouped.rename(
        columns={mse_col + "_mean": "mean", mse_col + "_sem": "sem"}, inplace=True
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = []
    values = []
    errors = []
    for _, row in grouped.iterrows():
        alg = str(row[alg_col])
        batch = f", B={int(row[batch_col])}" if batch_col and not pd.isna(row[batch_col]) else ""
        labels.append(_alg_style(alg)["label"] + batch)
        values.append(row["mean"])
        errors.append(row["sem"] if pd.notna(row["sem"]) else 0.0)
    y = np.arange(len(labels))
    ax.barh(y, values, xerr=errors, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("reconstruction MSE")
    ax.set_title("CIFAR-10 reconstruction MSE")
    ax.grid(alpha=0.25, axis="x")
    return [_savefig(fig, out_dir, "cifar_recon_mse.png")]


def _plot_cifar_images(results_dir: Path, out_dir: Path) -> List[Path]:
    """Plot reconstruction images if ``run_cifar.py`` saved an NPZ bundle."""
    import glob

    candidates = sorted(Path(results_dir).glob("cifar_reconstructions*.npz"))
    if not candidates:
        print("[plot_results] no CIFAR reconstruction image bundle found")
        return []

    written: List[Path] = []
    for path in candidates:
        try:
            bundle = np.load(path, allow_pickle=False)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[plot_results] failed to load {path}: {exc}")
            continue

        keys = [k for k in bundle.files if k.startswith("x") or k.startswith("recon")]
        keys = keys[:12]
        if not keys:
            continue

        n = len(keys)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows), squeeze=False
        )
        for idx, key in enumerate(keys):
            arr = bundle[key]
            arr = np.asarray(arr)
            if arr.ndim == 3:
                arr = arr.reshape(-1)
            # Assume flattened CIFAR image (3072); reshape as channels-last.
            if arr.size == 3072:
                arr = arr.reshape(32, 32, 3)
            ax = axes[idx // ncols][idx % ncols]
            ax.imshow(np.clip(arr, 0.0, 1.0) if arr.max() <= 1.0 else np.clip(arr / 255.0, 0.0, 1.0))
            ax.axis("off")
            ax.set_title(key, fontsize=7)
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")
        written.append(_savefig(fig, out_dir, f"{path.stem}.png"))
    return written


def plot_cifar(
    csv_path: Path,
    out_dir: Path,
    results_dir: Optional[Path] = None,
) -> List[Path]:
    """Plot CIFAR-10 reconstruction MSE and, if available, images."""
    written = _plot_cifar_mse(csv_path, out_dir)
    written.extend(_plot_cifar_images(results_dir or Path(csv_path).parent, out_dir))
    return written


# ---------------------------------------------------------------------------
# Appendix E.1: wallclock timings
# ---------------------------------------------------------------------------

def plot_wallclock_timings(
    results_dir: Path,
    out_dir: Path,
) -> List[Path]:
    """Aggregate wallclock timings across all available result CSVs."""
    results_dir = Path(results_dir)
    written: List[Path] = []

    for csv_path in sorted(results_dir.glob("*.csv")):
        if "wallclock" in csv_path.name:
            continue
        df = _read_csv(csv_path)
        if df is None:
            continue
        alg_col = _get_col(df, ["algorithm", "algo", "method"])
        time_col = _get_col(df, ["wallclock_seconds", "wallclock", "time"])
        if alg_col is None or time_col is None:
            continue

        grouped = (
            df.groupby(alg_col, as_index=False)[time_col]
            .agg(["mean", "sem"])
            .reset_index()
        )
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = [
                "_".join(str(c) for c in col).strip("_") for col in grouped.columns
            ]
        grouped.rename(
            columns={time_col + "_mean": "mean", time_col + "_sem": "sem"},
            inplace=True,
        )

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        algs = []
        means = []
        sems = []
        for _, row in grouped.iterrows():
            alg = str(row[alg_col])
            algs.append(_alg_style(alg)["label"])
            means.append(row["mean"])
            sems.append(row["sem"] if pd.notna(row["sem"]) else 0.0)
        y = np.arange(len(algs))
        ax.barh(y, means, xerr=sems, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(algs)
        ax.invert_yaxis()
        ax.set_xlabel("wallclock time (s)")
        ax.set_xscale("log")
        ax.set_title(f"wallclock timings: {csv_path.stem}")
        ax.grid(alpha=0.25, axis="x")
        written.append(_savefig(fig, out_dir, f"{csv_path.stem}_wallclock.png"))

    return written


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BaM reproduction figures from result CSVs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing experiment result CSVs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory in which to write figures",
    )
    parser.add_argument(
        "--figures",
        type=str,
        default="all",
        help="Comma-separated subset: gaussian,sinh_arcsinh,posteriordb,cifar,wallclock",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = set(args.figures.split(",")) if args.figures != "all" else None
    all_figures = {"gaussian", "sinh_arcsinh", "posteriordb", "cifar", "wallclock"}
    if figures is None:
        figures = all_figures

    if "gaussian" in figures:
        plot_gaussian_targets(results_dir / "gaussian_targets.csv", out_dir)
    if "sinh_arcsinh" in figures:
        plot_sinh_arcsinh(results_dir / "sinh_arcsinh.csv", out_dir)
    if "posteriordb" in figures:
        plot_posteriordb(results_dir / "posteriordb.csv", out_dir)
    if "cifar" in figures:
        plot_cifar(results_dir / "cifar.csv", out_dir, results_dir=results_dir)
    if "wallclock" in figures:
        plot_wallclock_timings(results_dir, out_dir)

    print("[plot_results] done.")


if __name__ == "__main__":
    main()
