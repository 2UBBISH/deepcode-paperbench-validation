"""Qualitative evaluation for *Stochastic Interpolants with Data-Dependent Couplings*.

This module reproduces the visual panels of the paper:

* **Figure 3 / Section 1** (in-painting, 256x256 and 512x512): each row is a triple
  ``(x_0, X_{t=1}, x_1)`` where the left panel is the *base distribution sample*
  ``x_0 = xi o x_1 + (1 - xi) o zeta`` (colorful static in the masked region), the
  middle panel is the *model sample* ``X_{t=1}`` obtained by integrating the
  probability-flow ODE (Eq. 8), and the right panel is the ground truth.
* **Figure 4** (super-resolution 64x64 -> 256x256) and **Figure 6**
  (256x256 -> 512x512): each row is the triple ``(U(D(x_1)), X_{t=1}, x_1)`` - the
  low-resolution image, the model sample ``X_{t=1}`` and the high-resolution
  ground truth.
* **Figure 5**: additional in-filling examples on 256x256 images "with temporal
  slices of the probability flow", i.e. snapshots ``X_{t_i}`` of the ODE
  trajectory between ``t = 0`` and ``t = 1``.

Images follow the repository-wide convention of living in ``[-1, 1]``; they are
mapped back to ``[0, 1]`` only for display.

The module is dependency-light: ``torch``/``numpy`` are used for array handling
and ``matplotlib`` is imported lazily inside the saving functions so that
training-time imports stay cheap.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

__all__ = [
    "IMAGE_INTERPOLATION",
    "TRIPLE_COLUMN_TITLES",
    "TripleFigure",
    "make_triples",
    "inpainting_triple",
    "superres_triple",
    "make_probability_flow_figure",
    "save_triples_grid",
    "save_probability_flow_grid",
    "save_image_grid",
    "to_display",
    "from_display",
    "denormalize",
    "masked_pixels",
    "verify_inpainting_consistency",
    "main",
]

logger = logging.getLogger(__name__)

#: Interpolation used when writing images to disk.
IMAGE_INTERPOLATION = "nearest"

#: Column titles used for the base/model/ground-truth triple grids (Figs. 3, 4, 6).
TRIPLE_COLUMN_TITLES = ("Base sample $x_0$", "Model sample $X_{t=1}$", "Ground truth")

# Column titles for super-resolution triplets (Fig. 4 / Fig. 6).
SUPERRES_COLUMN_TITLES = ("Low resolution", "Model sample $X_{t=1}$", "Ground truth")


# --------------------------------------------------------------------------------------
# tensor helpers
# --------------------------------------------------------------------------------------
def _as_tensor(x: Union[torch.Tensor, np.ndarray, Sequence[Any]]) -> torch.Tensor:
    """Convert ``x`` to a ``float32`` CPU tensor."""
    if isinstance(x, torch.Tensor):
        out = x.detach().float().cpu()
    else:
        out = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    return out


def _as_image_batch(x: Union[torch.Tensor, np.ndarray], ndim: int = 4) -> torch.Tensor:
    """Return images as a 4D ``(B, C, H, W)`` tensor (accepts 3D single images)."""
    t = _as_tensor(x)
    if t.ndim == ndim:
        return t
    if t.ndim == ndim - 1:
        return t.unsqueeze(0)
    # Tolerate (1, B, ...) style nesting by squeezing leading singleton dims.
    while t.ndim > ndim and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim != ndim:
        raise ValueError(f"expected {ndim - 1}D or {ndim}D image tensor, got shape {tuple(t.shape)}")
    return t


def to_display(images: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Map images from ``[-1, 1]`` to ``[0, 1]`` and return a numpy array ``(B, H, W, C)``."""
    t = _as_image_batch(images)
    t = ((t.clamp(-1.0, 1.0) + 1.0) / 2.0).permute(0, 2, 3, 1)
    return t.numpy()


def from_display(images: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Inverse of :func:`to_display`: ``[0, 1]`` ``(B, H, W, C)`` or ``(B, C, H, W)`` -> ``[-1, 1]``."""
    t = _as_tensor(images)
    if t.ndim == 4 and t.shape[-1] in (1, 3, 4) and t.shape[1] not in (1, 3, 4):
        t = t.permute(0, 3, 1, 2)
    return t.clamp(0.0, 1.0) * 2.0 - 1.0


def denormalize(images: Union[torch.Tensor, np.ndarray], mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """``x_std = (x - mean) / std`` back to ``[0, 1]`` display numpy arrays ``(B, H, W, C)``.

    The repository's canonical convention (``si/data/transforms.py``) is ``[-1, 1]``,
    i.e. ``mean = std = 0.5``; this helper simply makes that explicit.
    """
    t = _as_image_batch(images)
    t = t * std + mean
    return ((t.clamp(0.0, 1.0)).permute(0, 2, 3, 1)).numpy()


def masked_pixels(image: Union[torch.Tensor, np.ndarray], xi: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Return the pixels of ``image`` lying in the *missing* region (``xi == 0``)."""
    img = _as_image_batch(image)
    mask = _as_image_batch(xi, ndim=4)
    if mask.shape[1] == 1 and img.shape[1] > 1:
        mask = mask.expand_as(img)
    elif mask.shape[1] != img.shape[1]:
        mask = mask.expand(-1, img.shape[1], -1, -1)
    return img * (1.0 - mask)


def verify_inpainting_consistency(
    base: Union[torch.Tensor, np.ndarray],
    model_sample: Union[torch.Tensor, np.ndarray],
    ground_truth: Union[torch.Tensor, np.ndarray],
    xi: Union[torch.Tensor, np.ndarray],
    atol: float = 1e-4,
) -> Dict[str, float]:
    """Check the structural facts of Section 4.1 that the figures illustrate.

    For the in-painting coupling ``x_0 = xi o x_1 + (1 - xi) o zeta`` we have
    ``xi o I_t = xi o x_1`` for every ``t``, hence the observed pixels of the base
    sample, of the model sample and of the ground truth must all coincide.

    Returns a dictionary with the maximum absolute deviations on the observed
    (``xi == 1``) pixels and a boolean ``consistent`` flag.
    """
    x0 = _as_image_batch(base)
    x1 = _as_image_batch(model_sample)
    gt = _as_image_batch(ground_truth)
    mask = _as_image_batch(xi, ndim=4)
    if mask.shape[1] == 1 and gt.shape[1] > 1:
        mask = mask.expand_as(gt)
    elif mask.shape[1] != gt.shape[1]:
        mask = mask.expand(-1, gt.shape[1], -1, -1)

    dev_base = float(((x0 - gt).abs() * mask).max()) if x0.numel() else 0.0
    dev_model = float(((x1 - gt).abs() * mask).max()) if x1.numel() else 0.0
    return {
        "base_vs_gt_observed_max_abs": dev_base,
        "model_vs_gt_observed_max_abs": dev_model,
        "consistent": bool(dev_base <= atol and dev_model <= atol),
    }


# --------------------------------------------------------------------------------------
# triple container
# --------------------------------------------------------------------------------------
@dataclass
class TripleFigure:
    """One row of a qualitative figure: base/conditioning panel, model panel, ground truth.

    Attributes
    ----------
    base:
        For in-painting the base distribution sample ``x_0`` (with colorful static in the
        masked region); for super-resolution the low-resolution image ``U(D(x_1))``.
    model:
        The model sample ``X_{t=1}`` obtained by integrating the probability-flow ODE (8).
    ground_truth:
        The target image ``x_1``.
    xi:
        Optional conditioning tensor (missingness mask for in-painting, upsampled low-res
        image for super-resolution) stored for bookkeeping / validation.
    class_label:
        Optional ImageNet class label used to generate the sample.
    caption:
        Optional free-form caption (e.g. ``"ImageNet val #1234 (class 207)"``).
    extra:
        Any additional metadata (task, resolution, step counts, ...).
    """

    base: torch.Tensor
    model: torch.Tensor
    ground_truth: torch.Tensor
    xi: Optional[torch.Tensor] = None
    class_label: Optional[int] = None
    caption: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base = _as_image_batch(self.base)
        self.model = _as_image_batch(self.model)
        self.ground_truth = _as_image_batch(self.ground_truth)
        if self.xi is not None:
            self.xi = _as_image_batch(self.xi)
        if not (self.base.shape == self.model.shape == self.ground_truth.shape):
            raise ValueError(
                "base/model/ground_truth must share the same shape, got "
                f"{tuple(self.base.shape)}, {tuple(self.model.shape)}, {tuple(self.ground_truth.shape)}"
            )

    # ------------------------------------------------------------------ properties
    @property
    def resolution(self) -> Tuple[int, int]:
        """Spatial resolution ``(H, W)`` of the images."""
        return int(self.ground_truth.shape[-2]), int(self.ground_truth.shape[-1])

    @property
    def num_images(self) -> int:
        """Number of stacked images in this row."""
        return int(self.ground_truth.shape[0])

    @property
    def panels(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The ``(base, model, ground truth)`` panels."""
        return self.base, self.model, self.ground_truth

    def consistency(self, atol: float = 1e-4) -> Dict[str, float]:
        """Validate Section 4.1 structure when ``xi`` is a binary missingness mask."""
        if self.xi is None:
            return {}
        return verify_inpainting_consistency(self.base, self.model, self.ground_truth, self.xi, atol=atol)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"TripleFigure(shape={tuple(self.ground_truth.shape)}, "
            f"class_label={self.class_label}, caption={self.caption!r})"
        )


# --------------------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------------------
def make_triples(
    base: Union[torch.Tensor, np.ndarray],
    model: Union[torch.Tensor, np.ndarray],
    ground_truth: Union[torch.Tensor, np.ndarray],
    xi: Optional[Union[torch.Tensor, np.ndarray]] = None,
    class_labels: Optional[Sequence[int]] = None,
    captions: Optional[Sequence[str]] = None,
    **extra: Any,
) -> List[TripleFigure]:
    """Build one :class:`TripleFigure` per image from stacked panel tensors.

    Parameters
    ----------
    base, model, ground_truth:
        Tensors of shape ``(B, C, H, W)`` (or ``(C, H, W)`` for a single image) holding
        respectively the base sample / conditioning image, the model sample ``X_{t=1}``
        and the ground truth ``x_1``.
    xi:
        Optional conditioning tensor with the same batch/space layout in which the mask is
        ``1`` on observed and ``0`` on missing pixels (Section 4.1), or the upsampled
        low-resolution image for super-resolution (Section 4.2).
    class_labels:
        Optional ImageNet class labels, one per image.
    captions:
        Optional captions, one per image.

    Returns
    -------
    list of :class:`TripleFigure`
    """
    b = _as_image_batch(base)
    m = _as_image_batch(model)
    g = _as_image_batch(ground_truth)
    if not (b.shape == m.shape == g.shape):
        raise ValueError(
            f"panel shapes must match, got {tuple(b.shape)}, {tuple(m.shape)}, {tuple(g.shape)}"
        )
    x = _as_image_batch(xi) if xi is not None else None
    if x is not None and x.shape[0] != g.shape[0]:
        x = x.expand(g.shape[0], *x.shape[1:])

    n = g.shape[0]
    triples: List[TripleFigure] = []
    for i in range(n):
        triples.append(
            TripleFigure(
                base=b[i : i + 1],
                model=m[i : i + 1],
                ground_truth=g[i : i + 1],
                xi=None if x is None else x[i : i + 1],
                class_label=None if class_labels is None else int(class_labels[i]),
                caption=None if captions is None else captions[i],
                extra=dict(extra),
            )
        )
    return triples


def inpainting_triple(
    x0: Union[torch.Tensor, np.ndarray],
    model_sample: Union[torch.Tensor, np.ndarray],
    x1: Union[torch.Tensor, np.ndarray],
    xi: Optional[Union[torch.Tensor, np.ndarray]] = None,
    **kwargs: Any,
) -> List[TripleFigure]:
    """Convenience wrapper building in-painting triples (Figure 3 / Figure 5 layout)."""
    extra = dict(kwargs.pop("extra", {}) or {})
    extra["task"] = "inpainting"
    return make_triples(x0, model_sample, x1, xi=xi, extra=extra, **kwargs)


def superres_triple(
    low_res: Union[torch.Tensor, np.ndarray],
    model_sample: Union[torch.Tensor, np.ndarray],
    high_res: Union[torch.Tensor, np.ndarray],
    xi: Optional[Union[torch.Tensor, np.ndarray]] = None,
    **kwargs: Any,
) -> List[TripleFigure]:
    """Convenience wrapper building super-resolution triples (Figure 4 / Figure 6 layout)."""
    extra = dict(kwargs.pop("extra", {}) or {})
    extra["task"] = "superres"
    return make_triples(low_res, model_sample, high_res, xi=xi, extra=extra, **kwargs)


# --------------------------------------------------------------------------------------
# figure assembly
# --------------------------------------------------------------------------------------
def _import_pyplot():
    """Import matplotlib lazily with a headless-friendly backend."""
    import matplotlib

    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"), force=False)
    import matplotlib.pyplot as plt

    return plt


def _row_panels(
    triple: TripleFigure,
    columns: Sequence[Any],
) -> List[Any]:
    """Select the panels to display for one triple according to ``columns``."""
    base, model, gt = triple.panels
    lookup = {"base": base, "model": model, "ground_truth": gt, "low_res": base, "high_res": gt}
    panels = []
    for col in columns:
        if callable(col):
            panels.append(col(triple))
            continue
        key = str(col)
        if key not in lookup:
            raise ValueError(f"unknown column {key!r}; expected one of {sorted(lookup)}")
        panels.append(lookup[key])
    return panels


def save_triples_grid(
    triples: Sequence[TripleFigure],
    path: str,
    columns: Sequence[Any] = ("base", "model", "ground_truth"),
    column_titles: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    max_rows: Optional[int] = None,
    figsize_per_panel: Tuple[float, float] = (2.6, 2.6),
    fontsize: int = 9,
    dpi: int = 150,
    show_captions: bool = True,
) -> str:
    """Write a qualitative grid (one triple per row) such as Figures 3, 4 and 6.

    Parameters
    ----------
    triples:
        Rows of the figure.
    path:
        Output image path (``.png``/``.pdf``/...).
    columns:
        Which panels to show; defaults to ``(base, model, ground_truth)``.  For the
        low-resolution panel of super-resolution, ``"base"`` is the low-res image
        (Figure 4/6 use the low-resolution image as the left panel).
    column_titles:
        Overrides the header row; defaults to :data:`TRIPLE_COLUMN_TITLES`.
    title:
        Optional suptitle.
    max_rows:
        Optionally limit the number of rows drawn.

    Returns
    -------
    str
        The path that was written.
    """
    plt = _import_pyplot()

    rows = list(triples)
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    if not rows:
        raise ValueError("save_triples_grid requires at least one triple")

    if column_titles is None:
        column_titles = TRIPLE_COLUMN_TITLES

    n_rows = len(rows)
    n_cols = len(columns)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
        dpi=dpi,
    )

    for r, triple in enumerate(rows):
        panels = _row_panels(triple, columns)
        for c, panel in enumerate(panels):
            ax = axes[r][c]
            arr = to_display(panel)[0]
            ax.imshow(arr, interpolation=IMAGE_INTERPOLATION)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0 and c < len(column_titles):  # column headers on the first row
                ax.set_title(column_titles[c], fontsize=fontsize, pad=6)
        if show_captions and triple.caption:
            axes[r][0].set_ylabel(triple.caption, fontsize=max(fontsize - 2, 5), labelpad=4)
        elif show_captions and triple.class_label is not None:
            axes[r][0].set_ylabel(f"class {triple.class_label}", fontsize=max(fontsize - 2, 5), labelpad=4)

    if title:
        fig.suptitle(title, fontsize=fontsize + 2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote qualitative triple grid with %d rows to %s", n_rows, path)
    return path


def make_probability_flow_figure(
    trajectory: Union[torch.Tensor, np.ndarray],
    ground_truth: Optional[Union[torch.Tensor, np.ndarray]] = None,
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    times: Optional[Sequence[float]] = None,
    path: Optional[str] = None,
    title: Optional[str] = None,
    figsize_per_panel: Tuple[float, float] = (2.2, 2.2),
    fontsize: int = 8,
    dpi: int = 150,
) -> str:
    """Visualise temporal slices of the probability flow (Figure 5).

    Parameters
    ----------
    trajectory:
        Snapshots ``X_{t_i}`` of the ODE trajectory, shaped ``(T, C, H, W)``,
        ``(T, B, C, H, W)`` or ``(B, T, C, H, W)``.
    ground_truth:
        Optional target image shown as the last panel.
    mask:
        Optional in-painting mask; when given the observed pixels are patched back into
        every snapshot, which makes the "unmasked pixels stay fixed" property visible.
    times:
        Optional times label for each snapshot (defaults to a linear grid over ``[0, 1]``).
    path:
        Output path.  If ``None`` the figure is returned as a matplotlib ``Figure``
        object instead of being written... (in that case the returned value is the
        ``Figure`` object; otherwise the written path).

    Returns
    -------
    str or matplotlib.figure.Figure
    """
    plt = _import_pyplot()

    traj = _as_tensor(trajectory)
    if traj.ndim == 5:
        # (T, B, C, H, W) -> take the first sample of every snapshot
        if traj.shape[1] < traj.shape[0]:
            frames = traj[:, 0]
        else:
            frames = traj[0]  # (B, T, C, H, W) -> (T, C, H, W)
    elif traj.ndim == 4:
        frames = traj
    else:
        raise ValueError(f"trajectory must be 4D or 5D, got shape {tuple(traj.shape)}")

    if times is None:
        times = [i / max(len(frames) - 1, 1) for i in range(len(frames))]
    if len(times) != len(frames):
        raise ValueError(f"len(times)={len(times)} does not match {len(frames)} snapshots")

    gt = _as_image_batch(ground_truth)[0] if ground_truth is not None else None
    mk = _as_image_batch(mask)[0] if mask is not None else None
    if mk is not None and mk.shape[0] == 1 and frames.shape[1] > 1:
        mk = mk.expand(frames.shape[1], *mk.shape[1:])

    panels: List[torch.Tensor] = []
    labels: List[str] = []
    for i, frame in enumerate(frames):
        f = frame if frame.ndim == 4 else frame.unsqueeze(0)
        if mk is not None and gt is not None:
            # patch observed pixels back in (they are constant along the flow)
            m = mk.unsqueeze(0) if mk.ndim == 3 else mk
            f = f * (1.0 - m) + gt.unsqueeze(0) * m
        panels.append(f)
        labels.append(f"$t={times[i]:.2f}$")
    if gt is not None:
        panels.append(gt.unsqueeze(0))
        labels.append("Ground truth")

    n_cols = min(len(panels), 6)
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
        dpi=dpi,
    )
    for idx, panel in enumerate(panels):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.imshow(to_display(panel)[0], interpolation=IMAGE_INTERPOLATION)
        ax.set_title(labels[idx], fontsize=fontsize)
        ax.set_xticks([])
        ax.set_yticks([])
    for idx in range(len(panels), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    if title:
        fig.suptitle(title, fontsize=fontsize + 2)
    fig.tight_layout()

    if path is None:
        return fig
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote probability-flow figure with %d snapshots to %s", len(panels), path)
    return path


def save_probability_flow_grid(
    trajectories: Sequence[Union[torch.Tensor, np.ndarray]],
    path: str,
    ground_truth: Optional[Union[torch.Tensor, np.ndarray]] = None,
    masks: Optional[Sequence[Optional[Union[torch.Tensor, np.ndarray]]]] = None,
    **kwargs: Any,
) -> str:
    """Save several probability-flow rows (one per trajectory) into a single figure.

    Thin convenience wrapper used by ``evaluate.py`` when assembling Figure 5-style
    panels for many validation images at once.
    """
    plt = _import_pyplot()
    figs = []
    for i, traj in enumerate(trajectories):
        gt = None
        if ground_truth is not None:
            g = _as_image_batch(ground_truth)
            gt = g[i : i + 1]
        mk = None
        if masks is not None and masks[i] is not None:
            mk = masks[i]
        figs.append(make_probability_flow_figure(traj, ground_truth=gt, mask=mk, path=None, **kwargs))

    n = len(figs)
    fig, axes = plt.subplots(n, 1, figsize=(figs[0].get_size_inches()[0], sum(f.get_size_inches()[1] for f in figs)), dpi=150)
    if n == 1:
        axes = [axes]
    for ax, f in zip(np.atleast_1d(axes).ravel(), figs):
        f.canvas.draw()
        buf = np.asarray(f.canvas.buffer_rgba())
        ax.imshow(buf)
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    for f in figs:
        plt.close(f)
    plt.close(fig)
    return path


def save_image_grid(
    images: Union[torch.Tensor, np.ndarray],
    path: str,
    nrow: int = 8,
    title: Optional[str] = None,
    dpi: int = 150,
) -> str:
    """Save a plain grid of images (handy for unconditional sample dumps)."""
    plt = _import_pyplot()
    imgs = to_display(images)
    b = imgs.shape[0]
    nrow = max(1, min(int(nrow), b))
    ncol = int(np.ceil(b / nrow))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.0 * nrow), squeeze=False, dpi=dpi)
    for i, ax in enumerate(axes.ravel()):
        if i < b:
            ax.imshow(imgs[i], interpolation=IMAGE_INTERPOLATION)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# CLI: turn a saved panel bundle into a figure
# --------------------------------------------------------------------------------------
def _load_bundle(path: str) -> Dict[str, Any]:
    """Load a ``.pt``/``.npz``/``.pth`` bundle of figure panels.

    Accepted keys (``base``/``x0``, ``model``/``sample``/``x1_hat``,
    ``ground_truth``/``gt``/``x1``; optional ``xi``, ``trajectory``, ``times``).
    """
    if path.endswith(".npz"):
        with np.load(path) as data:
            return {k: data[k] for k in data.files}
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"unsupported bundle contents in {path}: {type(obj)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``python -m eval.qualitative --panels panels.pt --out fig3.png ...``"""
    parser = argparse.ArgumentParser(description="Qualitative triple / probability-flow figures")
    parser.add_argument("--panels", required=True, help=".pt or .npz bundle with base/model/ground_truth panels")
    parser.add_argument("--out", required=True, help="output image path")
    parser.add_argument("--task", default="inpainting", choices=["inpainting", "superres"])
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--flow", action="store_true", help="render temporal slices of the probability flow (Fig. 5)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    bundle = _load_bundle(args.panels)

    def pick(*names):
        for n in names:
            if n in bundle:
                return bundle[n]
        return None

    if args.flow:
        traj = pick("trajectory", "traj", "frames")
        if traj is None:
            raise SystemExit("--flow requires a 'trajectory' entry in the bundle")
        path = make_probability_flow_figure(
            traj,
            ground_truth=pick("ground_truth", "gt", "x1"),
            mask=pick("xi", "mask"),
            times=pick("times", None),
            path=args.out,
            title=args.title,
        )
    else:
        base = pick("base", "x0", "low_res")
        model = pick("model", "sample", "x1_hat")
        gt = pick("ground_truth", "gt", "x1", "high_res")
        if base is None or model is None or gt is None:
            raise SystemExit("bundle needs base/x0, model/sample and ground_truth/x1 entries")
        triples = make_triples(base, model, gt, xi=pick("xi", "mask", "cond"))
        titles = TRIPLE_COLUMN_TITLES if args.task == "inpainting" else SUPERRES_COLUMN_TITLES
        path = save_triples_grid(
            triples, args.out, column_titles=titles, title=args.title, max_rows=args.max_rows
        )
    print(path)
    return 0


def _self_test() -> None:  # pragma: no cover - manual smoke test
    torch.manual_seed(0)
    x1 = torch.rand(3, 3, 32, 32) * 2 - 1
    xi = (torch.rand(3, 1, 32, 32) > 0.3).float()
    zeta = torch.randn(3, 3, 32, 32)
    x0 = xi * x1 + (1 - xi) * zeta
    fake_model = x1 + 0.05 * torch.randn_like(x1)
    triples = inpainting_triple(x0, fake_model, x1, xi=xi)
    assert len(triples) == 3
    assert triples[0].consistency()["consistent"]
    save_triples_grid(triples, "/tmp/qual_triples.png")

    traj = torch.stack([i / 7 * x1 + (1 - i / 7) * x0 for i in range(8)])
    make_probability_flow_figure(traj, ground_truth=x1, mask=xi, path="/tmp/qual_flow.png")
    print("qualitative self-test OK")


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) > 1:
        raise SystemExit(main())
    _self_test()
