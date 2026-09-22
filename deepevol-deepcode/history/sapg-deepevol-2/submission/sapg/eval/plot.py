"""Plotting utilities to reproduce the SAPG paper figures.

This module consumes the JSON artifacts produced by the experiment runners
(``experiments/run_ppo_baseline.py``, ``experiments/run_sapg.py``,
``experiments/run_ablations.py``, ``experiments/run_reconstruction.py``) and
renders the paper's figures:

* Figure 2  -- PPO batch-size saturation (with optional SAPG dashed overlay).
* Figure 6  -- Aggregation ablations (leader vs symmetric vs none).
* Figure 8  -- L2 reconstruction error vs network size.

All functions are defensive: they accept either a path to a JSON file or an
already-loaded dict, and they degrade gracefully when optional keys are missing
so that partial runs can still be visualised.

Usage
-----
    python -m eval.plot --figure 2 --ppo runs/ppo_baseline/ppo_baseline_summary.json
    python -m eval.plot --figure 6 --ablation runs/ablations/ablation_summary.json
    python -m eval.plot --figure 8 --reconstruction runs/reconstruction/reconstruction_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

# Use a non-interactive backend so plotting works on headless machines.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_json(source: Any) -> Dict[str, Any]:
    """Accept a path, a JSON string, or an already-loaded dict."""
    if source is None:
        return {}
    if isinstance(source, dict):
        return source
    if isinstance(source, (str, bytes, os.PathLike)):
        path = os.fspath(source)
        if os.path.exists(path):
            with open(path, "r") as fh:
                return json.load(fh)
        # Maybe it is a raw JSON string.
        try:
            return json.loads(path)
        except (ValueError, TypeError):
            raise FileNotFoundError(f"Could not load JSON from: {source!r}")
    raise TypeError(f"Unsupported source type: {type(source)!r}")


def _ensure_dir(path: str) -> str:
    """Create the parent directory of ``path`` if needed and return ``path``."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    return path


def _smooth(values: Sequence[float], window: int = 5) -> List[float]:
    """Simple centered moving-average smoothing (edge-padded)."""
    values = list(values)
    n = len(values)
    if window <= 1 or n == 0:
        return values
    half = window // 2
    out: List[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def _extract_curve(history: Dict[str, Any], key: str) -> Tuple[List[float], List[float]]:
    """Return (x, y) for a history dict with an ``iteration`` axis."""
    if not history:
        return [], []
    y = history.get(key)
    if y is None:
        return [], []
    x = history.get("iteration")
    if x is None or len(x) != len(y):
        x = list(range(len(y)))
    return list(x), list(y)


def _primary_metric_key(task: Optional[str]) -> str:
    """Successes-per-episode for hard tasks, reward otherwise."""
    if task in {"regrasping", "throw", "reorientation"}:
        return "mean_successes"
    return "mean_reward"


# ---------------------------------------------------------------------------
# Figure 2 -- PPO batch-size saturation
# ---------------------------------------------------------------------------
def plot_batch_size_saturation(
    ppo_summary: Any,
    sapg_summary: Any = None,
    task: Optional[str] = None,
    metric: Optional[str] = None,
    output_path: str = "figures/fig2_batch_size_saturation.png",
    title: Optional[str] = None,
) -> str:
    """Reproduce Figure 2: PPO performance vs batch size (num_envs).

    Parameters
    ----------
    ppo_summary:
        Output of ``run_ppo_baseline.run_sweep`` (or path to its JSON). Expected
        schema: ``{"results": {"<batch_size>": {"history": {...}, "summary": {...}}}}``.
    sapg_summary:
        Optional SAPG result used to draw the dashed red asymptote line. May be
        the output of ``run_sapg.run`` (``{"history": ..., "summary": ...}``) or a
        plain float giving the SAPG asymptotic value.
    task:
        Task name; selects the primary metric (successes vs reward).
    metric:
        Explicit history key to plot. Overrides ``task``-based selection.
    """
    ppo = _load_json(ppo_summary)
    metric = metric or _primary_metric_key(task or ppo.get("task"))

    results = ppo.get("results", ppo)
    if not isinstance(results, dict) or not results:
        raise ValueError("PPO summary contains no 'results' to plot.")

    batch_sizes: List[int] = []
    finals: List[float] = []
    bests: List[float] = []
    for key, payload in results.items():
        try:
            bs = int(key)
        except (TypeError, ValueError):
            continue
        summary = payload.get("summary", payload) if isinstance(payload, dict) else {}
        final = summary.get(f"final_{metric}")
        best = summary.get(f"best_{metric}")
        if final is None and best is None:
            # Fall back to the last value of the history curve.
            history = payload.get("history", {}) if isinstance(payload, dict) else {}
            _, curve = _extract_curve(history, metric)
            if curve:
                final = curve[-1]
                best = max(curve)
        if final is None:
            continue
        batch_sizes.append(bs)
        finals.append(float(final))
        bests.append(float(best if best is not None else final))

    if not batch_sizes:
        raise ValueError(f"No usable '{metric}' values found in PPO summary.")

    order = sorted(range(len(batch_sizes)), key=lambda i: batch_sizes[i])
    batch_sizes = [batch_sizes[i] for i in order]
    finals = [finals[i] for i in order]
    bests = [bests[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(batch_sizes, bests, marker="o", color="tab:blue", label="PPO (best)")
    ax.plot(batch_sizes, finals, marker="s", linestyle="--", color="tab:cyan",
            alpha=0.8, label="PPO (final)")

    # Optional SAPG asymptote overlay (dashed red line).
    sapg_value: Optional[float] = None
    if sapg_summary is not None:
        if isinstance(sapg_summary, (int, float)):
            sapg_value = float(sapg_summary)
        else:
            sapg = _load_json(sapg_summary)
            summary = sapg.get("summary", sapg)
            sapg_value = summary.get(f"best_{metric}")
            if sapg_value is None:
                sapg_value = summary.get(f"final_{metric}")
            if sapg_value is None:
                _, curve = _extract_curve(sapg.get("history", {}), metric)
                if curve:
                    sapg_value = max(curve)
            if sapg_value is not None:
                sapg_value = float(sapg_value)

    if sapg_value is not None:
        ax.axhline(sapg_value, color="tab:red", linestyle="--", linewidth=2,
                   label="SAPG")

    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_xlabel("Batch size (num_envs)")
    ax.set_ylabel("Successes per episode" if metric == "mean_successes"
                  else "Episode reward")
    ax.set_title(title or f"PPO batch-size saturation ({ppo.get('env_name', 'env')})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Figure 6 -- Aggregation ablations
# ---------------------------------------------------------------------------
def plot_ablations(
    ablation_summary: Any,
    task: Optional[str] = None,
    metric: Optional[str] = None,
    output_path: str = "figures/fig6_ablations.png",
    title: Optional[str] = None,
) -> str:
    """Reproduce Figure 6: leader vs symmetric vs none aggregation.

    Expected schema (from ``run_ablations.run``)::

        {"results": {"<config_name>": {"mode": ..., "num_workers": ...,
                                       "history": {...}, "summary": {...}}}}
    """
    data = _load_json(ablation_summary)
    metric = metric or _primary_metric_key(task or data.get("task"))
    results = data.get("results", data)
    if not isinstance(results, dict) or not results:
        raise ValueError("Ablation summary contains no 'results' to plot.")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"leader": "tab:blue", "symmetric": "tab:orange", "none": "tab:green"}
    plotted = 0

    for name, payload in results.items():
        if not isinstance(payload, dict):
            continue
        history = payload.get("history", {})
        x, y = _extract_curve(history, metric)
        if not y:
            continue
        mode = payload.get("mode") or name.split("_")[0]
        num_workers = payload.get("num_workers")
        label = f"{mode}" + (f" (w={num_workers})" if num_workers else "")
        color = colors.get(mode, None)
        ax.plot(x, _smooth(y, window=5), color=color, label=label)
        plotted += 1

    if plotted == 0:
        raise ValueError(f"No usable '{metric}' curves found in ablation summary.")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Successes per episode" if metric == "mean_successes"
                  else "Episode reward")
    ax.set_title(title or f"SAPG aggregation ablations ({data.get('env_name', 'env')})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Figure 8 -- L2 reconstruction error vs network size
# ---------------------------------------------------------------------------
def plot_reconstruction(
    reconstruction_summary: Any,
    output_path: str = "figures/fig8_reconstruction.png",
    title: Optional[str] = None,
) -> str:
    """Reproduce Figure 8: L2 reconstruction error vs network size.

    Expected schema (from ``run_reconstruction.run``)::

        {"results": {"<method>": {"sizes": [...], "errors": [...]}}}
    """
    data = _load_json(reconstruction_summary)
    results = data.get("results", data)
    if not isinstance(results, dict) or not results:
        raise ValueError("Reconstruction summary contains no 'results' to plot.")

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = 0
    for method, payload in results.items():
        if not isinstance(payload, dict):
            continue
        sizes = payload.get("sizes")
        errors = payload.get("errors")
        if not sizes or not errors:
            continue
        ax.plot(sizes, errors, marker="o", label=method)
        plotted += 1

    if plotted == 0:
        raise ValueError("No usable reconstruction curves found.")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Network size (units per layer)")
    ax.set_ylabel("L2 reconstruction error")
    ax.set_title(title or "L2 reconstruction error vs network size")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Convenience: plot all available artifacts in a directory
# ---------------------------------------------------------------------------
def plot_all(
    ppo_path: Optional[str] = None,
    sapg_path: Optional[str] = None,
    ablation_path: Optional[str] = None,
    reconstruction_path: Optional[str] = None,
    output_dir: str = "figures",
    task: Optional[str] = None,
) -> Dict[str, str]:
    """Render every figure whose source artifact is available."""
    os.makedirs(output_dir, exist_ok=True)
    produced: Dict[str, str] = {}

    if ppo_path and os.path.exists(ppo_path):
        produced["fig2"] = plot_batch_size_saturation(
            ppo_path,
            sapg_summary=sapg_path if (sapg_path and os.path.exists(sapg_path)) else None,
            task=task,
            output_path=os.path.join(output_dir, "fig2_batch_size_saturation.png"),
        )
    if ablation_path and os.path.exists(ablation_path):
        produced["fig6"] = plot_ablations(
            ablation_path,
            task=task,
            output_path=os.path.join(output_dir, "fig6_ablations.png"),
        )
    if reconstruction_path and os.path.exists(reconstruction_path):
        produced["fig8"] = plot_reconstruction(
            reconstruction_path,
            output_path=os.path.join(output_dir, "fig8_reconstruction.png"),
        )
    return produced


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SAPG paper figures.")
    parser.add_argument("--figure", type=str, default="all",
                        choices=["2", "6", "8", "all"],
                        help="Which figure to render.")
    parser.add_argument("--ppo", type=str, default=None,
                        help="Path to ppo_baseline_summary.json (Figure 2).")
    parser.add_argument("--sapg", type=str, default=None,
                        help="Optional path to sapg_summary.json (Figure 2 overlay).")
    parser.add_argument("--ablation", type=str, default=None,
                        help="Path to ablation_summary.json (Figure 6).")
    parser.add_argument("--reconstruction", type=str, default=None,
                        help="Path to reconstruction_summary.json (Figure 8).")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name used to pick the primary metric.")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Directory to write figures into.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.figure == "2":
        path = plot_batch_size_saturation(
            args.ppo, sapg_summary=args.sapg, task=args.task,
            output_path=os.path.join(args.output_dir, "fig2_batch_size_saturation.png"),
        )
        print(f"Wrote {path}")
    elif args.figure == "6":
        path = plot_ablations(
            args.ablation, task=args.task,
            output_path=os.path.join(args.output_dir, "fig6_ablations.png"),
        )
        print(f"Wrote {path}")
    elif args.figure == "8":
        path = plot_reconstruction(
            args.reconstruction,
            output_path=os.path.join(args.output_dir, "fig8_reconstruction.png"),
        )
        print(f"Wrote {path}")
    else:
        produced = plot_all(
            ppo_path=args.ppo, sapg_path=args.sapg,
            ablation_path=args.ablation, reconstruction_path=args.reconstruction,
            output_dir=args.output_dir, task=args.task,
        )
        if not produced:
            print("No artifacts found to plot.")
        for name, path in produced.items():
            print(f"Wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
