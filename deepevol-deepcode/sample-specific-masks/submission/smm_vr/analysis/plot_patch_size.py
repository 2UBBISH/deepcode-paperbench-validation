"""Patch-size study analysis for SMM (paper Sec. 5 "Impact of Patch Size", Figure 4).

The paper states:

    "As an important hyperparameter in SMM, number of Max-Pooling layers, ``l``, can
    vary, which means different patch sizes ``2**l``. Since the 5-layer mask generator
    neural network has at most 4 Max-Pooling layers, we examine the impact of patch
    sizes in ``{2**0, 2**1, 2**2, 2**3, 2**4}``. Results are shown in Figure 4. As the
    patch size increases, the accuracy of the SMM increases first, followed by a
    plateau or decline. This suggests that overly small patches may cause over-fitting,
    while overly large patch sizes could result in a loss of details in SMM. We thus
    have set the patch size to be 8 across all datasets."

This module consumes the JSON artifact written by
:func:`smm_vr.experiments.run_ablations.run_patch_size_study`
(``patch_size_curves.json`` by default) and produces:

* the Figure-4 style plot (one curve per dataset, x-axis = patch size ``2**l`` on a
  log2 scale, y-axis = mean top-1 accuracy in percent over the three seeds, error bars
  = std),
* a text table of the curves,
* a trend check reproducing the paper's "rises first, then plateaus or declines" claim
  and confirming that the adopted patch size of 8 is optimal or near-optimal.

The number of max-pooling layers ``l`` is bounded by the mask generator depth: the
5-layer generator used with ResNet-18/50 supports at most 4 pooling layers
(``l <= 4``), matching the studied range ``{2**0, ..., 2**4}``.

The module is import-safe: without ``matplotlib`` it still writes a JSON fallback and
all pure-python aggregation helpers keep working.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Constants (paper / config conventions)
# --------------------------------------------------------------------------------------

#: Max-pooling-layer counts studied in Figure 4.
PATCH_STUDY_L: Tuple[int, ...] = (0, 1, 2, 3, 4)

#: Corresponding patch sizes ``2**l``.
PATCH_SIZES: Tuple[int, ...] = tuple(2 ** int(l) for l in PATCH_STUDY_L)

#: Patch size adopted for every dataset in the main experiments (l = 3).
DEFAULT_PATCH_SIZE: int = 8
DEFAULT_L: int = 3

#: The 5-layer mask generator has at most 4 max-pooling layers (paper Sec. 5).
MAX_POOLING_LAYERS_5: int = 4

#: Datasets used for the patch-size study (Figure 4, ResNet-18).
PATCH_STUDY_DATASETS: Tuple[str, ...] = ("cifar10", "svhn", "flowers102", "eurosat")

BACKBONE: str = "resnet18"

#: Default artifact names shared with ``experiments/run_ablations.py``.
DEFAULT_CURVES_FILE: str = "patch_size_curves.json"
DEFAULT_OUTPUT_DIR: str = os.path.join("outputs", "figures")
DEFAULT_FIGURE_NAME: str = "figure4_patch_size.pdf"

DISPLAY_NAMES: Dict[str, str] = {
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "svhn": "SVHN",
    "gtsrb": "GTSRB",
    "flowers102": "Flowers102",
    "dtd": "DTD",
    "ucf101": "UCF101",
    "food101": "Food101",
    "sun397": "SUN397",
    "eurosat": "EuroSAT",
    "oxfordpets": "OxfordPets",
}


# --------------------------------------------------------------------------------------
# Data loading / normalisation
# --------------------------------------------------------------------------------------


def resolve_curves_path(curves: Optional[str] = None, output_dir: Optional[str] = None) -> str:
    """Resolve the ``patch_size_curves.json`` path.

    Precedence: explicit ``curves`` path, then ``output_dir`` (with the artifact name
    produced by the ablation runner), then the ablation runner's own helper when
    importable, and finally the default relative path.
    """
    if curves:
        return curves
    if output_dir:
        return os.path.join(output_dir, DEFAULT_CURVES_FILE)
    try:  # pragma: no cover - depends on the runner being implemented
        from ..experiments.run_ablations import patch_size_curve_path  # type: ignore

        return patch_size_curve_path(None)
    except Exception:
        pass
    return os.path.join("outputs", DEFAULT_CURVES_FILE)


def load_curves(path: str) -> Dict[str, Any]:
    """Load the patch-size curve artifact, returning ``{}`` when it is missing."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected patch-size curve payload in {path!r}")
    return payload


def normalise_curves(payload: Any) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Normalise any reasonable curve payload to ``{dataset: {l: {mean, std}}}``.

    Accepted layouts (all encountered in practice when re-running the study):

    * ``{"curves": {dataset: {l: {mean, std}}}}`` - what the ablation runner writes
    * ``{"curves": {dataset: [{l, mean, std}, ...]}}``
    * ``{"l_values": [...], "curves": {dataset: {"mean": [...], "std": [...]}}}``
    * ``{dataset: {l: mean}}`` - a bare mapping of accuracies
    """
    if not isinstance(payload, dict):
        return {}

    curves = payload.get("curves", payload)
    if not isinstance(curves, dict):
        return {}

    # Layout with per-dataset accuracy lists aligned to a shared ``l_values`` list.
    if "l_values" in payload and any(
        isinstance(v, dict) and ("mean" in v or "accuracy" in v or "accuracies" in v)
        for v in curves.values()
    ):
        l_values = [int(l) for l in payload["l_values"]]
        out: Dict[str, Dict[int, Dict[str, float]]] = {}
        for dataset, entry in curves.items():
            means = list(entry.get("mean", []) or entry.get("accuracy", []) or entry.get("accuracies", []) or [])
            stds = list(entry.get("std", []) or [0.0] * len(means))
            out[str(dataset)] = {
                int(l): {
                    "mean": float(means[i]) if i < len(means) else float("nan"),
                    "std": float(stds[i]) if i < len(stds) else 0.0,
                }
                for i, l in enumerate(l_values)
            }
        return out

    out = {}
    for dataset, entry in curves.items():
        if isinstance(entry, dict):
            per_l: Dict[int, Dict[str, float]] = {}
            for key, value in entry.items():
                try:
                    l = int(key)
                except (TypeError, ValueError):
                    continue
                per_l[l] = _coerce_point(value)
            out[str(dataset)] = per_l
        elif isinstance(entry, (list, tuple)):
            per_l = {}
            for i, value in enumerate(entry):
                l = int(value.get("l", i)) if isinstance(value, dict) and "l" in value else i
                per_l[l] = _coerce_point(value)
            out[str(dataset)] = per_l
    return out


def _coerce_point(value: Any) -> Dict[str, float]:
    """Coerce one curve point into ``{"mean": ..., "std": ...}``."""
    if isinstance(value, dict):
        mean = value.get("mean", value.get("accuracy", value.get("acc", float("nan"))))
        std = value.get("std", value.get("stddev", 0.0))
        return {"mean": float(mean), "std": float(std or 0.0)}
    if isinstance(value, (list, tuple)):
        mean = float(value[0])
        std = float(value[1]) if len(value) > 1 else 0.0
        return {"mean": mean, "std": std}
    try:
        return {"mean": float(value), "std": 0.0}
    except (TypeError, ValueError):
        return {"mean": float("nan"), "std": 0.0}


def curves_to_table(
    curves: Dict[str, Dict[int, Dict[str, float]]],
    *,
    datasets: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[int, float]]:
    """Return ``{dataset: {l: mean_accuracy}}`` (a compact numeric view)."""
    names = list(datasets) if datasets else list(curves.keys())
    return {
        str(name): {
            int(l): float(point.get("mean", float("nan")))
            for l, point in sorted(curves.get(name, {}).items())
        }
        for name in names
        if name in curves
    }


# --------------------------------------------------------------------------------------
# Trend analysis (reproduces Figure 4's claim)
# --------------------------------------------------------------------------------------


def average_curve(
    curves: Dict[str, Dict[int, Dict[str, float]]],
    *,
    l_values: Sequence[int] = PATCH_STUDY_L,
) -> Dict[int, float]:
    """Unweighted average accuracy per ``l`` across the studied datasets."""
    out: Dict[int, float] = {}
    for l in [int(l) for l in l_values]:
        values = [
            float(point[l]["mean"])
            for point in curves.values()
            if l in point and not math.isnan(float(point[l]["mean"]))
        ]
        out[l] = sum(values) / len(values) if values else float("nan")
    return out


def best_l(
    curves: Dict[str, Dict[int, Dict[str, float]]],
    *,
    l_values: Sequence[int] = PATCH_STUDY_L,
) -> Optional[int]:
    """Return the ``l`` with the highest average accuracy (ties -> smallest ``l``)."""
    averages = average_curve(curves, l_values=l_values)
    usable = [(l, v) for l, v in averages.items() if not math.isnan(v)]
    if not usable:
        return None
    return max(usable, key=lambda item: (item[1], -item[0]))[0]


def check_patch_size_trend(
    curves: Dict[str, Dict[int, Dict[str, float]]],
    *,
    l_values: Sequence[int] = PATCH_STUDY_L,
    expected_l: int = DEFAULT_L,
    tolerance: float = 0.5,
    min_gain: float = 0.5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Check Figure 4's trend: accuracy rises first, then plateaus or declines.

    The paper reports that the accuracy first increases with ``l`` and then plateaus or
    declines, and that ``patch_size = 8`` (``l = 3``) was therefore adopted everywhere.

    Returns a report with ``averages`` (mean accuracy per ``l``), ``best_l`` /
    ``best_patch_size`` (empirical optimum), ``rising_to_peak``,
    ``peak_matches_default``, ``plateau_or_decline_after_peak`` and ``passed``.
    """
    averages = average_curve(curves, l_values=l_values)
    usable = [(l, v) for l, v in averages.items() if not math.isnan(v)]
    if not usable:
        return {
            "averages": averages,
            "best_l": None,
            "best_patch_size": None,
            "rising_to_peak": False,
            "peak_matches_default": False,
            "plateau_or_decline_after_peak": False,
            "passed": False,
            "note": "no curve data available",
        }

    peak_l, peak_value = max(usable, key=lambda item: (item[1], -item[0]))
    first_value = averages.get(min(l for l, _ in usable), float("nan"))
    rising_to_peak = bool(not math.isnan(first_value) and (peak_value - first_value) >= min_gain)

    default_value = averages.get(int(expected_l))
    peak_matches_default = bool(
        default_value is not None
        and not math.isnan(default_value)
        and (peak_value - default_value) <= tolerance
    )

    after = [v for l, v in usable if l > peak_l]
    plateau_or_decline_after_peak = bool(not after or max(after) <= peak_value + min_gain)

    report: Dict[str, Any] = {
        "averages": averages,
        "best_l": peak_l,
        "best_patch_size": 2 ** int(peak_l),
        "peak_accuracy": peak_value,
        "expected_l": int(expected_l),
        "expected_patch_size": 2 ** int(expected_l),
        "expected_accuracy": default_value,
        "rising_to_peak": rising_to_peak,
        "peak_matches_default": peak_matches_default,
        "plateau_or_decline_after_peak": plateau_or_decline_after_peak,
        "passed": bool(rising_to_peak and peak_matches_default and plateau_or_decline_after_peak),
    }
    if verbose:
        print(format_patch_size_table(curves, l_values=l_values))
        print(format_trend_report(report))
    return report


def per_dataset_trend(
    curves: Dict[str, Dict[int, Dict[str, float]]],
    *,
    l_values: Sequence[int] = PATCH_STUDY_L,
) -> Dict[str, bool]:
    """Per-dataset "rises then plateaus/declines" flags (peak is not at an endpoint)."""
    out: Dict[str, bool] = {}
    for dataset, points in curves.items():
        usable = [
            (int(l), float(points[l]["mean"]))
            for l in l_values
            if l in points and not math.isnan(float(points[l]["mean"]))
        ]
        if len(usable) < 2:
            out[str(dataset)] = False
            continue
        peak_l, peak_value = max(usable, key=lambda item: (item[1], -item[0]))
        first_l, first_value = min(usable, key=lambda item: item[0])
        last_value = max(usable, key=lambda item: item[0])[1]
        out[str(dataset)] = bool(
            peak_l not in (first_l, max(l for l, _ in usable))
            and peak_value > first_value
            and last_value <= peak_value
        )
    return out


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------


def format_patch_size_table(
    curves: Dict[str, Dict[int, Dict[str, float]]],
    *,
    l_values: Sequence[int] = PATCH_STUDY_L,
    datasets: Optional[Sequence[str]] = None,
    decimals: int = 1,
    include_average: bool = True,
) -> str:
    """Render a fixed-width table of mean +/- std accuracy per patch size."""
    l_values = [int(l) for l in l_values]
    names = [str(d) for d in (datasets if datasets is not None else curves.keys()) if d in curves]
    if not names:
        return "no patch-size curve data available"

    header = f"{'DATASET':<14}" + "".join(f"{('2^%d' % l):>16}" for l in l_values)
    lines = [header, "-" * len(header)]
    for name in names:
        row = f"{DISPLAY_NAMES.get(name, name):<14}"
        for l in l_values:
            point = curves[name].get(l)
            if point is None or math.isnan(float(point.get("mean", float("nan")))):
                row += f"{'-':>16}"
            else:
                cell = "%.*f+-%.*f" % (decimals, point["mean"], decimals, point.get("std", 0.0))
                row += f"{cell:>16}"
        lines.append(row)
    if include_average:
        averages = average_curve(curves, l_values=l_values)
        row = f"{'AVERAGE':<14}"
        for l in l_values:
            value = averages.get(l, float("nan"))
            row += f"{'-':>16}" if math.isnan(value) else f"{value:>{16}.{decimals}f}"
        lines.append("-" * len(header))
        lines.append(row)
    return "\n".join(lines)


def format_trend_report(report: Dict[str, Any]) -> str:
    """Human-readable rendering of :func:`check_patch_size_trend`."""
    if report.get("best_l") is None:
        return "patch-size trend: no data"
    expected = report.get("expected_accuracy")
    expected_str = "n/a" if expected is None or (isinstance(expected, float) and math.isnan(expected)) else f"{expected:.2f}%"
    lines = [
        "patch-size trend (paper: accuracy rises first, then plateaus or declines; default patch size = 8):",
        f"  best l                : {report['best_l']} (patch size {report['best_patch_size']}, {report['peak_accuracy']:.2f}%)",
        f"  adopted l             : {report['expected_l']} (patch size {report['expected_patch_size']}, {expected_str})",
        f"  rising to peak        : {report['rising_to_peak']}",
        f"  peak matches adopted  : {report['peak_matches_default']}",
        f"  plateau/decline after : {report['plateau_or_decline_after_peak']}",
        f"  PASSED                : {report['passed']}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Plotting (Figure 4)
# --------------------------------------------------------------------------------------


def _get_matplotlib():
    """Import matplotlib with a headless-safe backend, or return ``None``."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        return plt
    except Exception:  # pragma: no cover - matplotlib is optional
        return None


def plot_patch_size_curves(
    curves: Dict[str, Dict[int, Dict[str, float]]],
    *,
    output_path: Optional[str] = None,
    datasets: Optional[Sequence[str]] = None,
    l_values: Sequence[int] = PATCH_STUDY_L,
    title: str = "Impact of patch size $2^{l}$ (ResNet-18)",
    show_std: bool = True,
    show_average: bool = True,
    dpi: int = 200,
    figsize: Tuple[float, float] = (6.0, 4.5),
) -> Optional[str]:
    """Plot accuracy vs. patch size ``2**l`` (Figure 4) and optionally save it.

    Returns the saved path, or -- when ``matplotlib`` is unavailable -- the path of a
    JSON fallback written next to ``output_path`` (``None`` if nothing was requested).
    """
    l_values = [int(l) for l in l_values]
    names = [str(d) for d in (datasets if datasets is not None else curves.keys()) if d in curves]

    plt = _get_matplotlib()
    if plt is None:  # pragma: no cover - exercised only without matplotlib
        if output_path:
            fallback = os.path.splitext(output_path)[0] + ".json"
            os.makedirs(os.path.dirname(os.path.abspath(fallback)) or ".", exist_ok=True)
            with open(fallback, "w", encoding="utf-8") as handle:
                json.dump(curves_to_table(curves, datasets=names), handle, indent=2)
            return fallback
        return None

    fig, ax = plt.subplots(figsize=figsize)
    xs = [2 ** int(l) for l in l_values]

    for name in names:
        points = curves[name]
        ys = [float(points[l]["mean"]) if l in points else float("nan") for l in l_values]
        errs = (
            [float(points[l].get("std", 0.0)) if l in points else 0.0 for l in l_values]
            if show_std
            else None
        )
        ax.errorbar(
            xs,
            ys,
            yerr=errs,
            marker="o",
            markersize=4,
            linewidth=1.4,
            capsize=2.5,
            label=DISPLAY_NAMES.get(name, name),
        )

    if show_average:
        averages = average_curve(curves, l_values=l_values)
        ax.plot(
            xs,
            [averages.get(l, float("nan")) for l in l_values],
            marker="s",
            linestyle="--",
            color="black",
            linewidth=1.8,
            markersize=4,
            label="Average",
        )

    # Mark the patch size adopted across all datasets (8 -> l = 3).
    low, high = ax.get_ylim()
    if DEFAULT_PATCH_SIZE in xs:
        ax.axvline(DEFAULT_PATCH_SIZE, color="grey", linestyle=":", linewidth=1.0)
        ax.annotate(
            "adopted patch size 8",
            xy=(DEFAULT_PATCH_SIZE, low + 0.02 * (high - low)),
            xytext=(DEFAULT_PATCH_SIZE, high),
            fontsize=7,
            color="grey",
            rotation=90,
            va="top",
            ha="left",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("patch size $2^{l}$")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    if names or show_average:
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout()

    saved: Optional[str] = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        saved = output_path
    plt.close(fig)
    return saved


def plot_patch_size_from_file(
    curves_path: Optional[str] = None,
    *,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    datasets: Optional[Sequence[str]] = PATCH_STUDY_DATASETS,
    l_values: Sequence[int] = PATCH_STUDY_L,
    show_std: bool = True,
    show_average: bool = True,
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load ``patch_size_curves.json``, verify the trend and plot Figure 4."""
    path = resolve_curves_path(curves_path, output_dir)
    payload = load_curves(path)
    curves = normalise_curves(payload)
    if datasets is not None:
        keep = {str(d) for d in datasets}
        selected = {k: v for k, v in curves.items() if k in keep}
        # Only restrict when the artifact actually uses the study datasets.
        if selected:
            curves = selected

    figure_path = None
    if save:
        target = output_path or os.path.join(output_dir or DEFAULT_OUTPUT_DIR, DEFAULT_FIGURE_NAME)
        figure_path = plot_patch_size_curves(
            curves,
            output_path=target,
            datasets=datasets,
            l_values=l_values,
            show_std=show_std,
            show_average=show_average,
        )

    report = check_patch_size_trend(curves, l_values=l_values, verbose=False)
    table = format_patch_size_table(curves, l_values=l_values, datasets=datasets)

    if verbose:
        print(f"curves file : {path}")
        print(table)
        print(format_trend_report(report))
        if figure_path:
            print(f"figure      : {figure_path}")

    return {
        "curves_path": path,
        "figure_path": figure_path,
        "curves": curves,
        "table": table,
        "trend": report,
        "per_dataset_trend": per_dataset_trend(curves, l_values=l_values),
    }


# Alias matching the naming used by the other analysis/experiment entry points.
plot_figure4 = plot_patch_size_from_file


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the SMM patch-size study (Figure 4) from patch_size_curves.json",
    )
    parser.add_argument("--curves", default=None, help="path to patch_size_curves.json")
    parser.add_argument("--output", default=None, help="path of the output figure")
    parser.add_argument("--output-dir", default=None, help="directory holding the curves file / figure")
    parser.add_argument("--datasets", nargs="*", default=list(PATCH_STUDY_DATASETS))
    parser.add_argument("--l-values", nargs="*", type=int, default=list(PATCH_STUDY_L))
    parser.add_argument("--no-std", action="store_true", help="omit error bars")
    parser.add_argument("--no-average", action="store_true", help="omit the average curve")
    parser.add_argument("--no-save", action="store_true", help="analyse only, do not write the figure")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    plot_patch_size_from_file(
        curves_path=args.curves,
        output_path=args.output,
        output_dir=args.output_dir,
        datasets=args.datasets,
        l_values=args.l_values,
        show_std=not args.no_std,
        show_average=not args.no_average,
        save=not args.no_save,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PATCH_STUDY_L",
    "PATCH_SIZES",
    "DEFAULT_PATCH_SIZE",
    "DEFAULT_L",
    "MAX_POOLING_LAYERS_5",
    "PATCH_STUDY_DATASETS",
    "DEFAULT_CURVES_FILE",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_FIGURE_NAME",
    "resolve_curves_path",
    "load_curves",
    "normalise_curves",
    "curves_to_table",
    "average_curve",
    "best_l",
    "check_patch_size_trend",
    "per_dataset_trend",
    "format_patch_size_table",
    "format_trend_report",
    "plot_patch_size_curves",
    "plot_patch_size_from_file",
    "plot_figure4",
    "build_arg_parser",
    "main",
]
