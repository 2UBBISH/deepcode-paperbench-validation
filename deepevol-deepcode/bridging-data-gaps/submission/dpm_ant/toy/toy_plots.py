"""Plotting utilities for the Section 5.1 toy 2-D Gaussian experiment.

This module renders reproductions of Figure 2 of *Adapting Pretrained Diffusion
Models for Few-Shot Image Generation* (DPMs-ANT)::

    Figure 2(a): gradient-direction comparison between the 10,000-sample
                 reference, the baseline DDPM, ``DPMs-ANT w/o AN`` and the full
                 ``DPMs-ANT`` model (10 target samples repeated 1,000x), plus
                 the worst-case adversarial noise cloud and the resulting
                 ellipse / mean shift.
    Figure 2(b): heat-map of generated samples with the x-axis being the
                 diffusion timestep and the y-axis the sampled 2-D value
                 (projected onto the (1, 1) direction).
    Figure 2(c): same heat-map for the ANT-adapted model.

The module is intentionally defensive: ``matplotlib`` is imported lazily so the
rest of the reproduction keeps working in head-less / plotting-free
environments, and every function degrades to writing a ``.pt``/``.json`` dump
when a plotting backend is unavailable.

Everything here is *reporting only* -- no algorithm is defined in this file.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

LOGGER = logging.getLogger("dpm_ant.toy_plots")

__all__ = [
    "FigureStyle",
    "plot_figure2",
    "plot_gradient_directions",
    "plot_noise_cloud",
    "plot_heatmap",
    "plot_training_curves",
    "plot_loss_curves",
    "save_figure",
    "save_results_json",
]

# ---------------------------------------------------------------------------
# lazy matplotlib helpers
# ---------------------------------------------------------------------------


def _get_pyplot():
    """Return ``matplotlib.pyplot`` with a non-interactive backend, or ``None``."""
    try:  # pragma: no cover - environment dependent
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning("matplotlib unavailable (%s) - figures will be dumped as JSON/PT", exc)
        return None


def _get_numpy():
    try:  # pragma: no cover - environment dependent
        import numpy as np

        return np
    except Exception:  # pragma: no cover - environment dependent
        return None


def _as_numpy(x: Any):
    """Convert tensors / lists / tuples to numpy arrays (or nested lists)."""
    np = _get_numpy()
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    if np is None:  # pragma: no cover - fallback
        return x
    try:
        return np.asarray(x)
    except Exception:  # pragma: no cover
        return np.asarray(list(x))


def _ensure_dir(path: Optional[str]) -> Optional[str]:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------


class FigureStyle:
    """Small container for the consistent figure style used in Figure 2."""

    def __init__(
        self,
        figsize: Tuple[float, float] = (12.0, 4.0),
        dpi: int = 150,
        cmap: str = "viridis",
        grid: bool = True,
        reference_color: str = "tab:green",
        baseline_color: str = "tab:orange",
        wo_an_color: str = "tab:blue",
        ant_color: str = "tab:red",
        noise_color: str = "tab:purple",
        annotate: bool = True,
        title: bool = True,
    ) -> None:
        self.figsize = figsize
        self.dpi = dpi
        self.cmap = cmap
        self.grid = grid
        self.reference_color = reference_color
        self.baseline_color = baseline_color
        self.wo_an_color = wo_an_color
        self.ant_color = ant_color
        self.noise_color = noise_color
        self.annotate = annotate
        self.title = title

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **overrides) -> "FigureStyle":
        cfg = dict(cfg or {})
        cfg.update(overrides)
        kwargs = {k: v for k, v in cfg.items() if k in cls().__dict__}
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# result extraction helpers
# ---------------------------------------------------------------------------


def _find_key(results: Any, *names: str) -> Any:
    """Search a (possibly nested) results dict for the first matching key."""
    if isinstance(results, dict):
        for name in names:
            if name in results:
                return results[name]
        for value in results.values():
            found = _find_key(value, *names)
            if found is not None:
                return found
    return None


def _extract_gradient_table(results: Dict[str, Any]) -> List[Tuple[str, Optional[float], Optional[float]]]:
    """Build ``[(label, angle_deg, magnitude), ...]`` from toy results.

    Accepts either the dict produced by
    :func:`dpm_ant.toy.toy_2d.gradient_direction_experiment` or the enclosing
    ``run_toy_experiment`` payload.
    """
    table: List[Tuple[str, Optional[float], Optional[float]]] = []
    grad = _find_key(results, "gradient_direction", "gradient_directions", "gradients")
    if isinstance(grad, dict):
        for family in ("angles", "angle_deg", "degrees", "magnitudes", "norms"):
            sub = grad.get(family)
            if isinstance(sub, dict):
                for label, value in sub.items():
                    table.append((str(label), float(value) if family.startswith("angle") or family in ("degrees",) else None,
                                  float(value) if not (family.startswith("angle") or family in ("degrees",)) else None))
                break
        if not table:
            for label, value in grad.items():
                if isinstance(value, dict):
                    angle = value.get("angle_deg", value.get("angle"))
                    mag = value.get("magnitude", value.get("norm"))
                    table.append((str(label), _maybe_float(angle), _maybe_float(mag)))
                else:
                    table.append((str(label), _maybe_float(value), None))
    return [t for t in table if t[1] is not None or t[2] is not None]


def _maybe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _extract_heatmap(results: Any) -> Optional[Dict[str, Any]]:
    """Find a heat-map payload and normalise its keys."""
    payload = _find_key(results, "heatmap", "heatmaps")
    if not isinstance(payload, dict):
        return None
    out: Dict[str, Any] = {}
    for key in ("timesteps", "t", "steps"):
        if key in payload:
            out["timesteps"] = payload[key]
            break
    for key in ("values", "samples", "projection", "projections", "sampled_values"):
        if key in payload:
            out["values"] = payload[key]
            break
    for key in ("counts", "hist", "histogram"):
        if key in payload:
            out["counts"] = payload[key]
            break
    for extra in ("value_edges", "bin_edges", "edges"):
        if extra in payload:
            out["value_edges"] = payload[extra]
            break
    for extra in ("category", "generated", "kind"):
        if extra in payload:
            out["category"] = payload[extra]
            break
    return out or None


def _extract_noise_cloud(results: Any):
    """Return ``(points, mean, covariance)`` for the adversarial noise cloud."""
    cloud = _find_key(results, "noise_cloud", "adversarial_noise_cloud", "worst_case_noise")
    points = mean = cov = None
    if isinstance(cloud, dict):
        for key in ("noise", "noises", "points", "cloud", "samples", "worst_case"):
            if key in cloud:
                points = cloud[key]
                break
        mean = cloud.get("mean")
        cov = cloud.get("covariance", cloud.get("cov"))
        eigvecs = cloud.get("eigenvectors", cloud.get("principal_axes"))
        return points, mean, cov, eigvecs, cloud
    if cloud is not None:
        return cloud, None, None, None, {}
    stats = _find_key(results, "noise_cloud_statistics", "cloud_statistics")
    if isinstance(stats, dict):
        return None, stats.get("mean"), stats.get("covariance", stats.get("cov")), stats.get("principal_axes"), stats
    return None, None, None, None, {}


def _extract_deltas(results: Any) -> Optional[Tuple[float, float]]:
    """Extract the model-parameter gradient / mean-shift direction."""
    for key in ("parameter_gradient", "model_parameter_gradient", "gradient", "mean_shift"):
        value = _find_key(results, key)
        if value is not None:
            try:
                pair = [float(v) for v in list(value)[:2]]
                if len(pair) == 2:
                    return pair[0], pair[1]
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Figure 2(a)
# ---------------------------------------------------------------------------


def plot_gradient_directions(
    results: Optional[Dict[str, Any]] = None,
    ax: Any = None,
    style: Optional[FigureStyle] = None,
    out_path: Optional[str] = None,
    **kwargs,
):
    """Reproduce Figure 2(a): gradient directions for the four variants.

    Draws the reference (10,000-sample) gradient, the baseline DDPM gradient,
    the ``DPMs-ANT w/o AN`` gradient and the full ``DPMs-ANT`` gradient on a
    unit circle so the *direction* (the quantity the paper argues about) is the
    readable quantity; the radius encodes the gradient magnitude.
    """
    style = style or FigureStyle()
    results = results or {}
    plt = _get_pyplot()
    table = _extract_gradient_table(results)

    deltas = _extract_deltas(results)
    if not table and deltas is not None:
        table = [("ant", math.degrees(math.atan2(deltas[1], deltas[0])), math.hypot(*deltas))]

    labels_by_family = {
        "reference": ("reference (10k)", style.reference_color),
        "baseline": ("baseline DDPM", style.baseline_color),
        "wo_an": ("DPMs-ANT w/o AN", style.wo_an_color),
        "no_an": ("DPMs-ANT w/o AN", style.wo_an_color),
        "ant": ("DPMs-ANT", style.ant_color),
        "full": ("DPMs-ANT", style.ant_color),
        "source": ("source (pretrained)", "tab:gray"),
    }

    if plt is None or ax is None and plt is None:  # pragma: no cover - no backend
        return {"gradient_table": table, "path": None}

    own_fig = False
    if ax is None:
        fig = plt.figure(figsize=style.figsize, dpi=style.dpi)
        ax = fig.add_subplot(1, 1, 1)
        own_fig = True

    # unit circle helper
    theta = [i / 180.0 * math.pi for i in range(361)]
    ax.plot([math.cos(t) for t in theta], [math.sin(t) for t in theta],
            linestyle=":", color="0.7", linewidth=0.8, zorder=1)
    ax.axhline(0.0, color="0.85", linewidth=0.6, zorder=0)
    ax.axvline(0.0, color="0.85", linewidth=0.6, zorder=0)

    magnitudes = [t[2] for t in table if t[2] is not None] or [1.0]
    max_mag = max(magnitudes) if max(magnitudes) > 0 else 1.0

    for label, angle_deg, magnitude in table:
        if angle_deg is None:
            continue
        key = label.lower().replace(" ", "_").replace("-", "_")
        pretty, color = labels_by_family.get(key, (label, None))
        for family, (name, col) in labels_by_family.items():
            if family in key:
                pretty, color = name, col
                break
        mag = magnitude if magnitude not in (None, 0) else max_mag
        radius = mag / max_mag
        ang = math.radians(angle_deg)
        x, y = radius * math.cos(ang), radius * math.sin(ang)
        ax.annotate("", xy=(x, y), xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="->", color=color or "black", linewidth=2.0),
                    zorder=3)
        if style.annotate:
            ax.text(x * 1.06, y * 1.06, pretty, color=color or "black", fontsize=8, zorder=4)

    # reference direction dashed line for easy angular comparison
    for label, angle_deg, _mag in table:
        if "reference" in label.lower() and angle_deg is not None:
            ang = math.radians(angle_deg)
            ax.plot([0, math.cos(ang)], [0, math.sin(ang)],
                    linestyle="--", color=style.reference_color, alpha=0.35, zorder=2)
            break

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    if style.grid:
        ax.grid(alpha=0.3)
    ax.set_xlabel(r"$z_1$ / gradient component 1")
    ax.set_ylabel(r"$z_2$ / gradient component 2")
    if style.title:
        ax.set_title("(a) gradient direction of $\\nabla_\\theta \\mathcal{L}$")

    if table:
        summary = "\n".join(
            f"{lbl}: {ang:.1f}$^\\circ$" for lbl, ang, _m in table if ang is not None
        )
        ax.text(0.02, 0.02, summary, transform=ax.transAxes, fontsize=7,
                verticalalignment="bottom", bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    if own_fig and out_path:
        save_figure(plt.gcf(), out_path)
        plt.close(plt.gcf())

    return {"gradient_table": table, "path": out_path}


# ---------------------------------------------------------------------------
# noise cloud
# ---------------------------------------------------------------------------


def plot_noise_cloud(
    results: Optional[Dict[str, Any]] = None,
    ax: Any = None,
    style: Optional[FigureStyle] = None,
    out_path: Optional[str] = None,
    max_points: int = 2000,
    reference_noise: Any = None,
    **kwargs,
):
    """Plot the adversarial noise cloud (circle -> ellipse) for Figure 2(a).

    The paper notes that the adversarially selected noise cloud becomes an
    ellipse whose principal axis follows the model-parameter gradient rather
    than an isotropic circle.  Both the initial (and optional reference) Gaussian
    cloud and the final adversarial cloud are drawn together with 1-sigma
    ellipse overlays.
    """
    style = style or FigureStyle()
    results = results or {}
    plt = _get_pyplot()
    points, mean, cov, eigvecs, raw = _extract_noise_cloud(results)
    if reference_noise is None:
        reference_noise = _find_key(results, "reference_noise", "initial_noise", "noise_baseline")

    info: Dict[str, Any] = {
        "num_points": 0,
        "anisotropy": _maybe_float((raw or {}).get("anisotropy")),
        "principal_angle_deg": _maybe_float(
            (raw or {}).get("principal_angle_deg", (raw or {}).get("principal_axis_angle_deg"))
        ),
        "path": out_path,
    }
    if plt is None:
        return info

    own_fig = False
    if ax is None:
        fig = plt.figure(figsize=(style.figsize[1], style.figsize[1]), dpi=style.dpi)
        ax = fig.add_subplot(1, 1, 1)
        own_fig = True

    # isotropic reference cloud
    if reference_noise is not None:
        ref = _as_numpy(reference_noise)
        try:
            ref = ref.reshape(-1, 2)
            if ref.shape[0] > max_points:
                idx = torch.randperm(ref.shape[0])[:max_points].numpy() if _get_numpy() is None else None
                if idx is None:
                    ref = ref[:max_points]
                else:  # pragma: no cover
                    ref = ref[idx]
            ax.scatter(ref[:, 0], ref[:, 1], s=3, alpha=0.25, color="0.6",
                       label="isotropic noise", zorder=2)
        except Exception:
            pass

    if points is not None:
        arr = _as_numpy(points)
        try:
            arr = arr.reshape(-1, 2)
            if arr.shape[0] > max_points:
                step = max(1, arr.shape[0] // max_points)
                arr = arr[::step][:max_points]
            ax.scatter(arr[:, 0], arr[:, 1], s=4, alpha=0.4, color=style.noise_color,
                       label="adversarial noise", zorder=3)
            info["num_points"] = int(arr.shape[0])
        except Exception:
            arr = None

    # covariance ellipse
    if cov is not None:
        try:
            cov_np = _as_numpy(cov).reshape(2, 2)
            mean_np = _as_numpy(mean).reshape(2) if mean is not None else _as_numpy(0.0, ) if False else None
            if mean_np is None:
                mean_np = _as_numpy(mean).reshape(2) if mean is not None else None
            if mean_np is None:
                mean_np = _as_numpy(cov_np).sum() * 0.0
            _draw_covariance_ellipse(ax, mean_np, cov_np, color=style.noise_color, label="1-$\\sigma$ ellipse")
        except Exception:
            pass

    # principal axis from the model parameter gradient
    deltas = _extract_deltas(results)
    if deltas is not None:
        norm = math.hypot(*deltas) or 1.0
        scale = 1.5
        ax.annotate("", xy=(scale * deltas[0] / norm, scale * deltas[1] / norm),
                    xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="-|>", color=style.ant_color, linewidth=2.0),
                    zorder=5)
        if style.annotate:
            ax.text(scale * deltas[0] / norm, scale * deltas[1] / norm,
                    "model gradient", color=style.ant_color, fontsize=8, zorder=6)

    # mean shift arrow
    if mean is not None:
        try:
            mean_np = _as_numpy(mean).reshape(2)
            ax.annotate("", xy=(mean_np[0], mean_np[1]), xytext=(0.0, 0.0),
                        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.4),
                        zorder=4)
        except Exception:
            pass

    ax.set_aspect("equal")
    if style.grid:
        ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlabel(r"$\epsilon_1$")
    ax.set_ylabel(r"$\epsilon_2$")
    if style.title:
        ax.set_title("(a) worst-case noise cloud")

    if own_fig and out_path:
        save_figure(plt.gcf(), out_path)
        plt.close(plt.gcf())
    return info


def _draw_covariance_ellipse(ax, mean, cov, color="tab:purple", label=None, n_std=1.0):
    """Draw a covariance ellipse without depending on scipy/matplotlib patches."""
    import numpy as _np  # local: only used when numpy exists

    mean = _np.asarray(mean, dtype=float).reshape(2)
    cov = _np.asarray(cov, dtype=float).reshape(2, 2)
    vals, vecs = _np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = _np.clip(vals, 0.0, None)
    angle = _np.degrees(_np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * n_std * _np.sqrt(vals)
    theta = _np.linspace(0.0, 2.0 * _np.pi, 200)
    ellipse = _np.array([_np.cos(theta), _np.sin(theta)])
    transform = _np.array([[vecs[0, 0], vecs[0, 1]], [vecs[1, 0], vecs[1, 1]]]) @ _np.diag(
        [n_std * _np.sqrt(vals[0]), n_std * _np.sqrt(vals[1])]
    )
    pts = transform @ ellipse + mean[:, None]
    ax.plot(pts[0], pts[1], color=color, linewidth=1.6, label=label, zorder=4)
    return {"angle_deg": float(angle), "width": float(width), "height": float(height)}


# ---------------------------------------------------------------------------
# Figures 2(b) / 2(c)
# ---------------------------------------------------------------------------


def plot_heatmap(
    results_or_payload: Optional[Dict[str, Any]] = None,
    ax: Any = None,
    style: Optional[FigureStyle] = None,
    out_path: Optional[str] = None,
    xlabel: str = "diffusion timestep",
    ylabel: str = "sampled value (projected)",
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **kwargs,
):
    """Render the Figure 2(b)/(c) heat-map.

    The x-axis is the diffusion timestep and the y-axis is the sampled 2-D value
    (projected onto the model's gradient direction, per the addendum); colour is
    the histogram density of the 20,000 generated samples.
    """
    style = style or FigureStyle()
    plt = _get_pyplot()
    payload = _extract_heatmap(results_or_payload) or {}
    timesteps = payload.get("timesteps")
    values = payload.get("values")
    counts = payload.get("counts")
    edges = payload.get("value_edges")
    info: Dict[str, Any] = {"path": out_path, "num_timesteps": 0, "num_bins": 0}

    if plt is None:
        return info

    own_fig = False
    if ax is None:
        fig = plt.figure(figsize=(style.figsize[0] / 2.0, style.figsize[1]), dpi=style.dpi)
        ax = fig.add_subplot(1, 1, 1)
        own_fig = True

    plotted = False
    if counts is not None:
        arr = _as_numpy(counts)
        try:
            if arr.ndim == 2:
                extent = None
                if timesteps is not None and edges is not None:
                    ts = _as_numpy(timesteps).reshape(-1)
                    ed = _as_numpy(edges).reshape(-1)
                    if ts.size and ed.size:
                        dt = float(ts[1] - ts[0]) / 2.0 if ts.size > 1 else 0.5
                        extent = [float(ts[0]) - dt, float(ts[-1]) + dt, float(ed[0]), float(ed[-1])]
                im = ax.imshow(_transpose_for_plot(arr), aspect="auto", origin="lower",
                               cmap=style.cmap, extent=extent, vmin=vmin, vmax=vmax)
                if own_fig:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")
                info["num_bins"] = int(arr.shape[-1])
                plotted = True
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("heatmap imshow failed: %s", exc)

    if not plotted and values is not None and timesteps is not None:
        # fall back to a 2-D histogram of (t, projected value)
        try:
            vals = _as_numpy(values)
            ts = _as_numpy(timesteps).reshape(-1)
            if vals.ndim == 2 and vals.shape[1] == 2:
                # (N, 2) samples recorded at final step: plot value vs sample index
                ax.hist2d(vals[:, 1], vals[:, 0], bins=100, cmap=style.cmap)
                plotted = True
            elif vals.ndim == 1:
                hist, xedges, yedges = _hist2d(vals, ts, bins=80)
                ax.pcolormesh(xedges, yedges, hist.T, cmap=style.cmap, shading="auto",
                              vmin=vmin, vmax=vmax)
                plotted = True
            if plotted:
                info["num_bins"] = int(vals.shape[0])
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("heatmap fallback failed: %s", exc)

    if not plotted:
        ax.text(0.5, 0.5, "no heat-map data", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
    else:
        info["num_timesteps"] = int(len(_as_numpy(timesteps))) if timesteps is not None else 0

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if style.title and title:
        ax.set_title(title)

    if own_fig and out_path:
        save_figure(plt.gcf(), out_path)
        plt.close(plt.gcf())
    return info


def _transpose_for_plot(arr):
    """Orient a ``(timesteps, value_bins)`` histogram for ``imshow``."""
    try:
        if arr.shape[0] >= arr.shape[1]:
            return arr.T
    except Exception:  # pragma: no cover
        pass
    return arr


def _hist2d(values, timesteps, bins: int = 80):
    np = _get_numpy()
    values = _as_numpy(values).reshape(-1)
    timesteps = _as_numpy(timesteps).reshape(-1)
    if timesteps.size != values.size:  # reshape to a (steps, samples) grid
        n_steps = timesteps.size
        n_samples = values.size // max(n_steps, 1)
        if n_steps * n_samples == values.size:
            values = values.reshape(n_steps, n_samples)
            hist = []
            for row in values:
                h, edges = np.histogram(row, bins=bins)
                hist.append(h)
            info = np.asarray(hist, dtype=float)
            yedges = edges
            xedges = np.asarray(timesteps, dtype=float)
            return info, xedges, yedges
    hist, xedges, yedges = np.histogram2d(timesteps, values, bins=bins)
    return hist, xedges, yedges


# ---------------------------------------------------------------------------
# loss curves (supporting material)
# ---------------------------------------------------------------------------


def plot_training_curves(
    results: Optional[Dict[str, Any]] = None,
    history: Optional[Sequence[Dict[str, Any]]] = None,
    ax: Any = None,
    style: Optional[FigureStyle] = None,
    out_path: Optional[str] = None,
    **kwargs,
):
    """Plot source-pretraining / ANT loss curves if present in the results."""
    style = style or FigureStyle()
    plt = _get_pyplot()
    results = results or {}
    series: Dict[str, Sequence[float]] = {}

    hist = history if history is not None else _find_key(results, "history", "loss_history", "history_ant")
    if isinstance(hist, list) and hist and isinstance(hist[0], dict):
        for key in ("loss", "ant_loss", "total", "sg_loss", "inner_loss"):
            values = [h.get(key) for h in hist if isinstance(h, dict) and h.get(key) is not None]
            if values:
                series[key] = [float(v) for v in values]
    elif isinstance(hist, list) and hist and isinstance(hist[0], (int, float)):
        series["loss"] = [float(v) for v in hist]

    for name, values in (("pretrain_loss", _find_key(results, "pretrain_loss", "source_loss_history")) or (),):
        if values:
            series[name] = [float(v) for v in values]

    if plt is None or not series:
        return {"series": {k: len(v) for k, v in series.items()}, "path": out_path}

    own_fig = False
    if ax is None:
        fig = plt.figure(figsize=(style.figsize[0] / 3.0, style.figsize[1] * 0.75), dpi=style.dpi)
        ax = fig.add_subplot(1, 1, 1)
        own_fig = True

    for name, values in series.items():
        ax.plot(range(len(values)), values, label=name, linewidth=1.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    if style.grid:
        ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    if style.title:
        ax.set_title("training loss")

    if own_fig and out_path:
        save_figure(plt.gcf(), out_path)
        plt.close(plt.gcf())
    return {"series": {k: len(v) for k, v in series.items()}, "path": out_path}


#: alias kept for convenience / backwards compatibility
def plot_loss_curves(*args, **kwargs):
    """Alias of :func:`plot_training_curves`."""
    return plot_training_curves(*args, **kwargs)


# ---------------------------------------------------------------------------
# top-level Figure 2 assembly
# ---------------------------------------------------------------------------


def plot_figure2(
    results: Optional[Dict[str, Any]] = None,
    out_dir: Optional[str] = None,
    prefix: str = "figure2",
    style: Optional[FigureStyle] = None,
    dpi: Optional[int] = None,
    save_json: bool = True,
    reference_noise: Any = None,
    **kwargs,
) -> Dict[str, Any]:
    """Reproduce Figure 2 of the paper.

    Parameters
    ----------
    results:
        Either the payload returned by
        :func:`dpm_ant.toy.toy_2d.run_toy_experiment` (preferred) or the raw
        gradient / noise-cloud / heat-map dictionary.
    out_dir:
        Directory where ``figure2a.{png,pdf}``, ``figure2b.png`` and
        ``figure2c.png`` are written.  If ``None``, nothing is written to disk.
    prefix:
        Filename prefix for the generated files.
    style / dpi:
        Cosmetic overrides.
    save_json:
        Also dump the extracted numeric summary next to the figures.

    Returns
    -------
    dict
        Paths of the written files plus the extracted numeric summaries.
    """
    results = results or {}
    style = style or FigureStyle()
    if dpi:
        style.dpi = dpi
    out_dir = _ensure_dir(out_dir)

    report: Dict[str, Any] = {
        "out_dir": out_dir,
        "style": style.to_dict(),
        "gradients": None,
        "noise_cloud": None,
        "heatmap_baseline": None,
        "heatmap_ant": None,
        "files": {},
    }

    plt = _get_pyplot()
    if plt is None:
        LOGGER.warning("matplotlib unavailable; writing numeric summary only")
        report["gradients"] = {"gradient_table": _extract_gradient_table(results)}
        if save_json and out_dir:
            path = save_results_json(report, os.path.join(out_dir, f"{prefix}_summary.json"))
            report["files"]["summary"] = path
        return report

    # ---- Figure 2(a): gradient direction + noise cloud side by side ----------
    try:
        fig = plt.figure(figsize=(style.figsize[0], style.figsize[1]), dpi=style.dpi)
        ax_grad = fig.add_subplot(1, 2, 1)
        ax_cloud = fig.add_subplot(1, 2, 2)
        grad_info = plot_gradient_directions(results, ax=ax_grad, style=style)
        cloud_info = plot_noise_cloud(results, ax=ax_cloud, style=style, reference_noise=reference_noise)
        fig.tight_layout()
        if out_dir:
            path_a = os.path.join(out_dir, f"{prefix}a.png")
            save_figure(fig, path_a)
            report["files"]["figure2a"] = path_a
        plt.close(fig)
        report["gradients"] = grad_info
        report["noise_cloud"] = cloud_info
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Figure 2(a) rendering failed: %s", exc)

    # ---- Figures 2(b) and 2(c): heat-maps -----------------------------------
    baseline_payload = _find_key(results, "heatmap_baseline", "heatmap_ddpm", "baseline_heatmap")
    ant_payload = _find_key(results, "heatmap_ant", "heatmap", "heatmaps", "ant_heatmap")
    if baseline_payload is None and isinstance(results.get("heatmaps"), dict):
        heatmaps = results["heatmaps"]
        baseline_payload = heatmaps.get("baseline", heatmaps.get("ddpm"))
        ant_payload = heatmaps.get("ant", heatmaps.get("full"))
    if ant_payload is None:
        ant_payload = _extract_heatmap(results)

    for name, payload, panel in (
        ("b", baseline_payload, "(b) baseline DPM"),
        ("c", ant_payload, "(c) DPMs-ANT"),
    ):
        if payload is None:
            continue
        try:
            fig = plt.figure(figsize=(style.figsize[0] / 2.0, style.figsize[1]), dpi=style.dpi)
            ax = fig.add_subplot(1, 1, 1)
            info = plot_heatmap(payload, ax=ax, style=style,
                                title=panel if style.title else None)
            fig.tight_layout()
            if out_dir:
                path = os.path.join(out_dir, f"{prefix}{name}.png")
                save_figure(fig, path)
                report["files"][f"figure2{name}"] = path
            plt.close(fig)
            report["heatmap_baseline" if name == "b" else "heatmap_ant"] = info
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Figure 2(%s) rendering failed: %s", name, exc)

    # ---- optional loss curves ----------------------------------------------
    try:
        if _find_key(results, "history", "loss_history", "pretrain_loss") is not None:
            fig = plt.figure(figsize=(style.figsize[0] / 3.0, style.figsize[1] * 0.75), dpi=style.dpi)
            ax = fig.add_subplot(1, 1, 1)
            info = plot_training_curves(results, ax=ax, style=style)
            fig.tight_layout()
            if out_dir:
                path = os.path.join(out_dir, f"{prefix}_loss.png")
                save_figure(fig, path)
                report["files"]["loss_curves"] = path
            plt.close(fig)
            report["loss_curves"] = info
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.debug("loss curve rendering failed: %s", exc)

    if save_json and out_dir:
        report["files"]["summary"] = save_results_json(
            report, os.path.join(out_dir, f"{prefix}_summary.json")
        )
    return report


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def save_figure(fig, path: str, also_pdf: bool = True, dpi: Optional[int] = None) -> str:
    """Save a matplotlib figure, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    if also_pdf and path.lower().endswith(".png"):
        try:
            fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight", dpi=dpi)
        except Exception:  # pragma: no cover - optional format
            pass
    LOGGER.info("wrote %s", path)
    return path


def save_results_json(report: Dict[str, Any], path: str, indent: int = 2) -> str:
    """Dump a JSON-serialisable summary of the plotted quantities."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    def _default(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if hasattr(obj, "tolist"):
            try:
                return obj.tolist()
            except Exception:  # pragma: no cover
                pass
        return str(obj)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=indent, default=_default)
    LOGGER.info("wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI (convenience: re-plot an existing toy_results.json)
# ---------------------------------------------------------------------------


def build_arg_parser():  # pragma: no cover - CLI convenience
    import argparse

    parser = argparse.ArgumentParser(description="Plot Figure 2 of the DPMs-ANT toy experiment")
    parser.add_argument("--results", type=str, default=None,
                        help="toy_results.json produced by dpm_ant/toy/toy_2d.py")
    parser.add_argument("--out-dir", type=str, default="outputs/toy/figure2")
    parser.add_argument("--prefix", type=str, default="figure2")
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI convenience
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    payload: Dict[str, Any] = {}
    if args.results and os.path.exists(args.results):
        with open(args.results, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        LOGGER.warning("no results file provided/found; rendering an empty template")

    report = plot_figure2(payload, out_dir=args.out_dir, prefix=args.prefix,
                          dpi=args.dpi, save_json=not args.no_json)
    LOGGER.info("files: %s", json.dumps(report.get("files", {}), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
