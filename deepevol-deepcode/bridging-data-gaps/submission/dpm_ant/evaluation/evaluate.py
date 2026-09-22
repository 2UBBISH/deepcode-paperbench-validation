"""End-to-end evaluation pipeline for DPMs-ANT (paper: "Adapting Pretrained
Diffusion Models for Few-Shot Image Generation", §5.2 Evaluation Metrics, §5.3,
Appendix B.2/B.4, Tables 1-4, 8 and 9).

This module orchestrates the full quantitative/qualitative evaluation of an
*adapted* diffusion model:

1. **Sampling** — generate a batch of images with the adapted backbone using
   the reverse process of ``dpm_ant/sampling/sampler.py`` (DDIM ``eta=0`` with
   100 steps by default, DDPM ``eta=1`` / 1000 steps optional).  For LDMs the
   samples are drawn in the 64x64 latent space and decoded with the frozen
   autoencoder.
2. **Intra-LPIPS (diversity, higher is better)** — 1,000 generated images are
   assigned to the nearest 10-shot training image by LPIPS distance; pairwise
   LPIPS distances inside each cluster are averaged and then averaged across
   clusters (§5.2).  A model that duplicates training samples scores 0.
3. **FID (quality, lower is better)** — computed against the *larger* target
   datasets (Babies 2.7k, Sunglasses 2.5k) following DDPM-PA; 10-shot FID is
   disabled by default because the paper notes it is unstable.
4. **Efficiency metrics** — parameter rate, GPU memory and wall-clock time
   (Table 1/8, §5.3) via ``dpm_ant/evaluation/metrics.py``.
5. **Paper comparison** — the measured numbers are diffed against the
   numbers reported in Tables 1-4, 5-7 and 8 so a reproduction run can be
   validated automatically.

The module works on in-memory tensors, on directories of generated images, or
on a full model, and it is deliberately defensive: any missing optional
dependency (``lpips``, ``clean-fid``, a GPU, a dataset directory) degrades
gracefully instead of aborting the evaluation.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

LOGGER = logging.getLogger(__name__)

__all__ = [
    "PAPER_INTRA_LPIPS",
    "PAPER_FID",
    "PAPER_PARAM_RATE",
    "PAPER_ABLATION_FID",
    "PAPER_SENSITIVITY",
    "EvalConfig",
    "EvalResult",
    "Evaluator",
    "evaluate_model",
    "evaluate_from_dirs",
    "run_evaluation",
    "aggregate_seeds",
    "format_report",
    "save_report",
    "load_report",
    "compare_to_paper",
    "load_images_from_dir",
    "resolve_reference_dir",
]


# ---------------------------------------------------------------------------
# Paper reference tables (used for automatic comparison of a reproduction)
# ---------------------------------------------------------------------------
#: Table 1 (Intra-LPIPS, up) + Table 4 (Sketches / Amedeo). Values are
#: ``(mean, std)`` as reported in the paper.
PAPER_INTRA_LPIPS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "TGAN": {
        "babies": (0.510, 0.026),
        "sunglasses": (0.550, 0.021),
        "raphael": (0.533, 0.023),
        "haunted_houses": (0.585, 0.007),
        "landscape_drawings": (0.601, 0.030),
        "sketches": (0.394, 0.023),
        "amedeo": (0.548, 0.026),
    },
    "TGAN+ADA": {
        "babies": (0.546, 0.033),
        "sunglasses": (0.571, 0.034),
        "raphael": (0.546, 0.037),
        "haunted_houses": (0.615, 0.018),
        "landscape_drawings": (0.643, 0.060),
        "sketches": (0.427, 0.022),
        "amedeo": (0.560, 0.019),
    },
    "EWC": {
        "babies": (0.560, 0.019),
        "sunglasses": (0.550, 0.014),
        "raphael": (0.541, 0.023),
        "haunted_houses": (0.579, 0.035),
        "landscape_drawings": (0.596, 0.052),
        "sketches": (0.430, 0.018),
        "amedeo": (0.594, 0.028),
    },
    "CDC": {
        "babies": (0.583, 0.014),
        "sunglasses": (0.581, 0.011),
        "raphael": (0.564, 0.010),
        "haunted_houses": (0.620, 0.029),
        "landscape_drawings": (0.674, 0.024),
        "sketches": (0.454, 0.017),
        "amedeo": (0.620, 0.029),
    },
    "DCL": {
        "babies": (0.579, 0.018),
        "sunglasses": (0.574, 0.007),
        "raphael": (0.558, 0.033),
        "haunted_houses": (0.616, 0.043),
        "landscape_drawings": (0.626, 0.021),
        "sketches": (0.461, 0.021),
        "amedeo": (0.616, 0.043),
    },
    "DDPM-PA": {
        "babies": (0.599, 0.024),
        "sunglasses": (0.604, 0.014),
        "raphael": (0.581, 0.041),
        "haunted_houses": (0.628, 0.029),
        "landscape_drawings": (0.706, 0.030),
        "sketches": (0.495, 0.024),
        "amedeo": (0.626, 0.022),
    },
    "DDPM-ANT": {
        "babies": (0.592, 0.016),
        "sunglasses": (0.613, 0.023),
        "raphael": (0.621, 0.068),
        "haunted_houses": (0.648, 0.010),
        "landscape_drawings": (0.723, 0.020),
        "sketches": (0.544, 0.025),
        "amedeo": (0.620, 0.021),
    },
    "LDM-ANT": {
        "babies": (0.601, 0.018),
        "sunglasses": (0.613, 0.011),
        "raphael": (0.592, 0.048),
        "haunted_houses": (0.653, 0.010),
        "landscape_drawings": (0.738, 0.026),
    },
}

#: Table 2 (FID, down) on the larger target sets.
PAPER_FID: Dict[str, Dict[str, float]] = {
    "TGAN": {"babies": 104.79, "sunglasses": 55.61},
    "TGAN+ADA": {"babies": 102.58, "sunglasses": 53.64},
    "EWC": {"babies": 87.41, "sunglasses": 59.73},
    "CDC": {"babies": 74.39, "sunglasses": 42.13},
    "DCL": {"babies": 52.56, "sunglasses": 38.01},
    "DDPM-PA": {"babies": 48.92, "sunglasses": 34.75},
    "DDPM-ANT": {"babies": 46.70, "sunglasses": 20.06},
}

#: Table 1 "Parameter Rate" column / §5.3.
PAPER_PARAM_RATE: Dict[str, float] = {
    "TGAN": 1.0,
    "TGAN+ADA": 1.0,
    "EWC": 1.0,
    "CDC": 1.0,
    "DCL": 1.0,
    "DDPM-PA": 1.0,
    "DDPM-ANT": 0.013,
    "LDM-ANT": 0.016,
}

#: Figure 4 / §5.4 ablation FIDs (10-shot Sunglasses, 300 iterations).
PAPER_ABLATION_FID: Dict[str, float] = {
    "full_model_finetune": 41.88,
    "adaptor_only": 38.65,
    "ant_wo_an": 26.41,
    "full_ant": 20.66,
}

#: Appendix B.3 Tables 5-7 optimum values.
PAPER_SENSITIVITY: Dict[str, Any] = {
    "gamma": {"grid": [1, 3, 5, 7, 9], "best": 5.0},
    "omega": {"grid": [0.01, 0.02, 0.03, 0.04, 0.05], "best": 0.02},
    "iterations": {"grid": [0, 100, 200, 300, 400], "best": 300},
    "best_fid": 18.13,
}

#: §5.3 wall-clock reference (GPU hours).
PAPER_TIME_HOURS: Dict[str, float] = {"ant_300_iters": 3.0, "baseline_5000_iters": 4.2}

#: §5.5 classifier ablation (FFHQ -> Sunglasses).
PAPER_CLASSIFIER_ABLATION: Dict[str, Dict[str, float]] = {
    "10_shot_classifier": {"intra_lpips": 0.613, "intra_lpips_std": 0.023, "fid": 20.06},
    "100_shot_classifier": {"intra_lpips": 0.637, "intra_lpips_std": 0.013, "fid": 22.84},
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """Settings for an evaluation run.

    Defaults follow §5.2: 1,000 generated images for Intra-LPIPS, DDIM with
    100 steps, FID against the larger (2.5k/2.7k) target sets only.
    """

    backbone: str = "ddpm"
    task: Optional[str] = None

    # sampling
    method: str = "ddim"
    num_steps: int = 100
    eta: float = 0.0
    batch_size: int = 16
    num_samples: int = 1000
    guidance_scale: float = 0.0
    use_classifier_guidance: bool = False
    latent_size: Optional[int] = None
    latent_channels: Optional[int] = None
    seed: int = 0
    seeds: List[int] = field(default_factory=list)

    # metrics
    compute_intra_lpips: bool = True
    compute_fid: bool = True
    compute_10shot_fid: bool = False
    intra_lpips_num_samples: int = 1000
    lpips_net: str = "alex"
    include_singletons: bool = True
    cluster_average: str = "mean"
    fid_targets: List[str] = field(default_factory=lambda: ["sunglasses", "babies"])
    fid_backend: Optional[str] = None
    fid_batch_size: int = 32

    # efficiency
    compute_efficiency: bool = False
    train_iterations: Optional[int] = None
    benchmark_iters: int = 5

    # io
    out_dir: Optional[str] = None
    sample_dir: Optional[str] = None
    report_path: Optional[str] = None
    save_images: bool = False
    save_pt: bool = False
    device: Optional[str] = None
    rel_tol: float = 0.25
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not self.seeds:
            self.seeds = [int(self.seed)]
        if self.method.lower() in ("ddpm",) and self.eta == 0.0:
            # DDPM is the stochastic variant: eta = 1.
            self.eta = 1.0

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, backbone: Optional[str] = None,
                  task: Optional[str] = None, **overrides: Any) -> "EvalConfig":
        """Build from a nested config (``evaluation``/``sampling``/``diffusion``/``tasks``)."""
        cfg = cfg or {}
        flat: Dict[str, Any] = {}

        ev = cfg.get("evaluation", cfg)
        if isinstance(ev, dict):
            flat.update({k: v for k, v in ev.items() if not isinstance(v, dict)})
            ip = ev.get("intra_lpips")
            if isinstance(ip, dict):
                flat.setdefault("intra_lpips_num_samples", ip.get("num_samples", 1000))
                flat.setdefault("lpips_net", ip.get("lpips_net", "alex"))
                flat.setdefault("include_singletons", ip.get("include_singletons", True))
                flat.setdefault("cluster_average", ip.get("cluster_average", "mean"))
                flat.setdefault("compute_intra_lpips", ip.get("enabled", True))
            fd = ev.get("fid")
            if isinstance(fd, dict):
                flat.setdefault("fid_backend", fd.get("backend"))
                flat.setdefault("fid_batch_size", fd.get("batch_size", 32))
                flat.setdefault("compute_10shot_fid", fd.get("compute_10shot_fid",
                                                             fd.get("compute_10shot", False)))
                targets = fd.get("targets")
                if isinstance(targets, list) and targets:
                    flat.setdefault("fid_targets", targets)
                flat.setdefault("compute_fid", fd.get("enabled", True))
            mt = ev.get("metrics")
            if isinstance(mt, dict):
                flat.setdefault("compute_efficiency", mt.get("enabled", False))

        sm = cfg.get("sampling")
        if isinstance(sm, dict):
            flat.setdefault("method", sm.get("method", "ddim"))
            flat.setdefault("num_steps", sm.get("num_steps", 100))
            flat.setdefault("eta", sm.get("eta", 0.0))
            flat.setdefault("batch_size", sm.get("batch_size", 16))
            flat.setdefault("num_samples", sm.get("num_samples", 1000))
            flat.setdefault("use_classifier_guidance", sm.get("use_classifier_guidance", False))
            flat.setdefault("guidance_scale", sm.get("guidance_scale", 0.0))

        ds = cfg.get("data")
        if isinstance(ds, dict):
            flat.setdefault("latent_size", ds.get("ldm_latent_size"))
            flat.setdefault("latent_channels", ds.get("ldm_latent_channels"))

        if backbone is None:
            backbone = flat.get("backbone", cfg.get("backbone", "ddpm"))
        if task is not None and isinstance(cfg.get("tasks"), dict):
            task_cfg = cfg["tasks"].get(task, {})
            if isinstance(task_cfg, dict) and task_cfg.get("backbone"):
                backbone = task_cfg["backbone"]

        flat = {k: v for k, v in flat.items() if v is not None}
        flat.update({k: v for k, v in overrides.items() if v is not None})
        flat["backbone"] = backbone
        if task is not None:
            flat["task"] = task

        known = {f for f in cls.__dataclass_fields__}
        flat = {k: v for k, v in flat.items() if k in known}
        return cls(**flat)

    def replace(self, **overrides: Any) -> "EvalConfig":
        data = self.to_dict()
        data.update({k: v for k, v in overrides.items() if v is not None})
        return EvalConfig(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    """Aggregated evaluation result for a (method, task) pair."""

    method: str = "DPMs-ANT"
    backbone: str = "ddpm"
    task: Optional[str] = None
    intra_lpips: Optional[float] = None
    intra_lpips_std: Optional[float] = None
    intra_lpips_per_seed: List[float] = field(default_factory=list)
    fid: Dict[str, float] = field(default_factory=dict)
    fid_std: Dict[str, float] = field(default_factory=dict)
    parameter_rate: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    train_hours: Optional[float] = None
    num_generated: int = 0
    seeds: List[int] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------
def _torch_load_images(files: Sequence[str], size: Optional[int] = 256) -> torch.Tensor:
    """Load a list of image files as a ``[-1, 1]`` tensor ``(N, 3, H, W)``."""
    from .intra_lpips import load_reference_images  # local import (torch-heavy)

    if not files:
        raise ValueError("no image files to load")
    return load_reference_images.__wrapped__(files) if hasattr(
        load_reference_images, "__wrapped__"
    ) else _load_via_datasets(files, size=size)


def _load_via_datasets(files: Sequence[str], size: Optional[int] = 256) -> torch.Tensor:
    from ..data.datasets import load_image  # local import

    imgs = [load_image(f, size=size) for f in files]
    return torch.stack(imgs, dim=0)


def load_images_from_dir(root: str, size: Optional[int] = 256,
                         limit: Optional[int] = None,
                         recursive: bool = True) -> torch.Tensor:
    """Load all images under ``root`` as a single ``[-1, 1]`` tensor.

    Supports ``.npy``/``.pt`` tensor dumps produced by the sampler as well.
    """
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(f"generated-image directory not found: {root}")

    from ..data.datasets import list_images, load_image  # local import

    files = list_images(root, recursive=recursive)
    if limit is not None:
        files = files[: int(limit)]
    if not files:
        raise FileNotFoundError(f"no images found under {root}")

    imgs: List[torch.Tensor] = []
    for f in files:
        try:
            imgs.append(load_image(f, size=size))
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("failed to load %s: %s", f, exc)
    if not imgs:
        raise RuntimeError(f"failed to load any image from {root}")
    return torch.stack(imgs, dim=0)


def _config_dir(cfg: Optional[Dict[str, Any]], name: str,
                keys: Sequence[str]) -> Optional[str]:
    """Look up a dataset directory in a (possibly nested) config."""
    if not isinstance(cfg, dict):
        return None
    data = cfg.get("data", cfg)
    if not isinstance(data, dict):
        return None
    for container in ("target_dirs", "fid_target_dirs", "source_dirs", "dirs", "targets"):
        block = data.get(container)
        if isinstance(block, dict):
            for key in keys:
                if block.get(key):
                    return block[key]
            # case-insensitive / alias-loose match
            for k, v in block.items():
                if isinstance(k, str) and any(key in k.lower() for key in keys):
                    return v
    for key in keys:
        if isinstance(data.get(key), str):
            return data[key]
    return None


def resolve_reference_dir(target: Optional[str], cfg: Optional[Dict[str, Any]] = None,
                          shots: int = 10, backbone: str = "ddpm") -> Optional[str]:
    """Resolve the training-image directory of a 10-shot target dataset.

    Returns ``None`` when the directory is not configured (the caller should
    then pass explicit reference images).
    """
    if target is None:
        return None
    keys = [str(target).lower()]
    try:
        from ..data.datasets import DATASET_ALIASES, TARGET_DATASETS, resolve_split

        canonical = resolve_split(target)
        keys.append(canonical)
        meta = TARGET_DATASETS.get(canonical, {})
        keys.extend([str(a).lower() for a in meta.get("aliases", [])])
        # include any dataset whose name contains one of our keys
        for name in TARGET_DATASETS:
            if any(k in name for k in keys):
                keys.append(name)
        _ = DATASET_ALIASES
    except Exception:  # pragma: no cover - optional dependency
        pass

    keys = sorted({k for k in keys if k}, key=len, reverse=True)
    return _config_dir(cfg, target, keys)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class Evaluator:
    """Run generation + Intra-LPIPS + FID (+ efficiency) for one adapted model."""

    def __init__(self,
                 model: Optional[torch.nn.Module] = None,
                 classifier: Optional[torch.nn.Module] = None,
                 autoencoder: Optional[torch.nn.Module] = None,
                 config: Optional[EvalConfig] = None,
                 cfg: Optional[Dict[str, Any]] = None,
                 backbone: Optional[str] = None,
                 task: Optional[str] = None,
                 device: Optional[str] = None,
                 reference_dir: Optional[str] = None,
                 reference_images: Optional[torch.Tensor] = None,
                 **overrides: Any):
        self.cfg = config or EvalConfig.from_dict(cfg, backbone=backbone, task=task, **overrides)
        if device is not None:
            self.cfg.device = device
        if backbone is None:
            backbone = self.cfg.backbone
        self.cfg.backbone = backbone
        if task is not None:
            self.cfg.task = task

        self.model = model
        self.classifier = classifier
        self.autoencoder = autoencoder
        self.raw_cfg = cfg
        self.device = self.cfg.device or "cpu"

        self.reference_images = reference_images
        self.reference_dir = reference_dir or resolve_reference_dir(task, cfg)

        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()
        if self.classifier is not None and hasattr(self.classifier, "eval"):
            self.classifier.eval()

    # -- shapes -------------------------------------------------------------
    def sample_shape(self) -> Tuple[int, int, int, int]:
        """Return ``(C, H, W, 0)``-style shape used for latent/pixel sampling."""
        if self.cfg.backbone == "ldm":
            size = self.cfg.latent_size or getattr(self.autoencoder, "latent_size", 64) or 64
            ch = self.cfg.latent_channels or getattr(self.autoencoder, "latent_channels", 4) or 4
            return int(ch), int(size), int(size), 0
        return 3, 256, 256, 0

    # -- generation ---------------------------------------------------------
    def generate(self,
                 num_samples: Optional[int] = None,
                 num_steps: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 seed: Optional[int] = None,
                 return_latents: bool = False,
                 out_dir: Optional[str] = None,
                 progress: Optional[bool] = None) -> torch.Tensor:
        """Sample ``num_samples`` images from the adapted model (Eq. 2/3)."""
        if self.model is None:
            raise ValueError("Evaluator.generate requires a model")

        from ..sampling.sampler import build_sampler  # local import

        num_samples = int(num_samples or self.cfg.num_samples)
        num_steps = num_steps or self.cfg.num_steps
        batch_size = int(batch_size or self.cfg.batch_size)
        method = str(self.cfg.method).lower()
        eta = self.cfg.eta if self.cfg.eta is not None else (0.0 if method == "ddim" else 1.0)

        sampler = build_sampler(
            cfg=self.raw_cfg,
            model=self.model,
            classifier=self.classifier if self.cfg.use_classifier_guidance else None,
            autoencoder=self.autoencoder,
            backbone=self.cfg.backbone,
            task=self.cfg.task,
            device=self.device,
            method=method,
            num_steps=num_steps,
            eta=eta,
            batch_size=batch_size,
            num_samples=num_samples,
            guidance_scale=self.cfg.guidance_scale,
            seed=self.cfg.seed if seed is None else int(seed),
            clip_denoised=True,
        )
        verbose = self.cfg.verbose if progress is None else bool(progress)
        images = sampler.sample(
            batch_size=batch_size,
            num_samples=num_samples,
            seed=self.cfg.seed if seed is None else int(seed),
            verbose=verbose,
            return_latents=False,
        )
        if not torch.is_tensor(images):
            images = torch.as_tensor(images)
        images = images.detach().float().cpu()

        if out_dir is not None:
            self.save_samples(images, out_dir)
        if return_latents:
            return images
        return images

    # -- image saving -------------------------------------------------------
    @staticmethod
    def save_samples(images: torch.Tensor, out_dir: str,
                     prefix: str = "sample", save_pt: bool = False) -> List[str]:
        """Persist generated ``[-1, 1]`` tensors to disk (PNG if PIL available)."""
        os.makedirs(out_dir, exist_ok=True)
        paths: List[str] = []
        imgs = images.detach().float().cpu()
        imgs = imgs.clamp(-1.0, 1.0)
        if imgs.min() >= -1.0 and imgs.max() <= 1.0:
            imgs = (imgs + 1.0) / 2.0
        try:
            from PIL import Image  # type: ignore

            for i, img in enumerate(imgs):
                arr = (img.clamp(0, 1) * 255.0).round().to(torch.uint8)
                arr = arr.permute(1, 2, 0).numpy()
                path = os.path.join(out_dir, f"{prefix}_{i:05d}.png")
                Image.fromarray(arr).save(path)
                paths.append(path)
        except Exception as exc:  # pragma: no cover - PIL missing
            LOGGER.warning("PIL unavailable (%s); falling back to .pt dump", exc)
            path = os.path.join(out_dir, f"{prefix}.pt")
            torch.save(imgs.cpu(), path)
            paths.append(path)
        if save_pt:
            path = os.path.join(out_dir, f"{prefix}_all.pt")
            torch.save(imgs.cpu(), path)
            paths.append(path)
        return paths

    # -- Intra-LPIPS --------------------------------------------------------
    def intra_lpips(self,
                    generated: Optional[torch.Tensor] = None,
                    reference_images: Optional[torch.Tensor] = None,
                    reference_dir: Optional[str] = None,
                    return_details: bool = False) -> Dict[str, Any]:
        """Compute Intra-LPIPS (§5.2): nearest-training-sample clustering."""
        from .intra_lpips import compute_intra_lpips  # local import

        if generated is None:
            generated = self.generate(num_samples=self.cfg.intra_lpips_num_samples)

        ref = reference_images if reference_images is not None else self.reference_images
        ref_dir = reference_dir or self.reference_dir

        result = compute_intra_lpips(
            generated,
            train_images=ref,
            train_dir=ref_dir,
            num_samples=min(int(self.cfg.intra_lpips_num_samples), int(generated.shape[0])),
            net=self.cfg.lpips_net,
            device=self.device,
            batch_size=max(1, int(self.cfg.batch_size)),
            include_singletons=bool(self.cfg.include_singletons),
            cluster_average=self.cfg.cluster_average,
            return_details=return_details,
        )
        return result

    # -- FID ----------------------------------------------------------------
    def fid(self,
            generated: Optional[torch.Tensor] = None,
            targets: Optional[Sequence[str]] = None,
            reference_dir: Optional[str] = None,
            return_details: bool = False) -> Dict[str, Any]:
        """Compute FID against the larger target dataset(s) (Table 2)."""
        from .fid import compute_fid  # local import

        if generated is None:
            generated = self.generate()
        targets = list(targets or self.cfg.fid_targets)

        out: Dict[str, Any] = {}
        for target in targets:
            ref_dir = reference_dir or _config_dir(
                self.raw_cfg, target, [str(target).lower()]
            )
            if ref_dir is None and reference_dir is None:
                LOGGER.warning("no FID reference directory configured for target '%s'", target)
            try:
                res = compute_fid(
                    generated,
                    reference_dir=ref_dir,
                    cfg=self.raw_cfg,
                    device=self.device,
                    backend=self.cfg.fid_backend,
                    batch_size=self.cfg.fid_batch_size,
                    target=target,
                    return_details=return_details,
                )
                out[target] = res
                LOGGER.info("FID[%s] = %.4f (n_ref=%s)", target, res.get("fid", float("nan")),
                            res.get("num_reference"))
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("FID for target '%s' failed: %s", target, exc)
                out[target] = {"fid": float("nan"), "error": str(exc), "target": target}
        return out

    # -- efficiency ---------------------------------------------------------
    def efficiency(self, step_fn: Optional[Callable[..., Any]] = None,
                   iterations: Optional[int] = None) -> Dict[str, Any]:
        """Parameter rate / GPU memory / time report (§5.3, Table 8)."""
        from .metrics import efficiency_report  # local import

        variant = "DPMs+ANT" if self.cfg.backbone in ("ddpm", "ldm") else "DPMs"
        report = efficiency_report(
            model=self.model,
            cfg=self.raw_cfg,
            backbone=self.cfg.backbone,
            variant=variant,
            device=self.device,
            iterations=iterations or self.cfg.train_iterations,
            benchmark_iters=self.cfg.benchmark_iters,
            step_fn=step_fn,
            compare=True,
        )
        return report

    # -- full run -----------------------------------------------------------
    def evaluate(self,
                 num_samples: Optional[int] = None,
                 seed: Optional[int] = None,
                 compute_intra_lpips: Optional[bool] = None,
                 compute_fid: Optional[bool] = None,
                 return_details: bool = False,
                 out_dir: Optional[str] = None) -> Dict[str, Any]:
        """Generate once and compute every enabled metric for a single seed."""
        seed = self.cfg.seed if seed is None else int(seed)
        n = int(num_samples or self.cfg.num_samples)
        do_il = self.cfg.compute_intra_lpips if compute_intra_lpips is None else bool(compute_intra_lpips)
        do_fid = self.cfg.compute_fid if compute_fid is None else bool(compute_fid)

        out_dir = out_dir or self.cfg.sample_dir
        t0 = time.time()
        generated = self.generate(num_samples=n, seed=seed, out_dir=out_dir)
        gen_time = time.time() - t0

        report: Dict[str, Any] = {
            "method": "DPMs-ANT",
            "backbone": self.cfg.backbone,
            "task": self.cfg.task,
            "seed": seed,
            "num_generated": int(generated.shape[0]),
            "generation_seconds": gen_time,
            "sampling": {
                "method": self.cfg.method,
                "num_steps": self.cfg.num_steps,
                "eta": self.cfg.eta,
            },
        }

        if do_il:
            il = self.intra_lpips(generated, return_details=return_details)
            report["intra_lpips"] = il
        if do_fid:
            report["fid"] = self.fid(generated, return_details=return_details)

        if self.cfg.compute_efficiency:
            try:
                report["efficiency"] = self.efficiency()
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("efficiency measurement failed: %s", exc)

        if self.cfg.rel_tol is not None:
            report["comparison"] = compare_to_paper(report, backbone=self.cfg.backbone,
                                                    rel_tol=self.cfg.rel_tol)
        return report


# ---------------------------------------------------------------------------
# Seed aggregation / reporting
# ---------------------------------------------------------------------------
def aggregate_seeds(reports: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-seed reports into ``mean ± std`` summary (Tables 1/4 style)."""
    reports = [r for r in reports if isinstance(r, dict)]
    if not reports:
        return {}

    il_vals: List[float] = []
    fid_vals: Dict[str, List[float]] = {}
    for r in reports:
        il = r.get("intra_lpips")
        if isinstance(il, dict):
            v = il.get("intra_lpips")
        else:
            v = il
        if isinstance(v, (int, float)) and v == v:
            il_vals.append(float(v))
        fid = r.get("fid", {})
        if isinstance(fid, dict):
            for target, res in fid.items():
                val = res.get("fid") if isinstance(res, dict) else res
                if isinstance(val, (int, float)) and val == val:
                    fid_vals.setdefault(target, []).append(float(val))

    def _ms(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
        if not vals:
            return None, None
        t = torch.tensor(vals, dtype=torch.float64)
        if t.numel() == 1:
            return float(t.item()), 0.0
        return float(t.mean().item()), float(t.std(unbiased=True).item())

    il_m, il_s = _ms(il_vals)
    summary: Dict[str, Any] = {
        "num_seeds": len(reports),
        "seeds": [r.get("seed") for r in reports],
        "intra_lpips": il_m,
        "intra_lpips_std": il_s,
        "intra_lpips_per_seed": il_vals,
        "fid": {},
        "fid_std": {},
        "tasks": sorted({str(r.get("task")) for r in reports if r.get("task")}),
        "backbones": sorted({str(r.get("backbone")) for r in reports if r.get("backbone")}),
    }
    for target, vals in fid_vals.items():
        m, s = _ms(vals)
        summary["fid"][target] = m
        summary["fid_std"][target] = s

    eff = [r.get("efficiency") for r in reports if isinstance(r.get("efficiency"), dict)]
    if eff:
        rates = [e.get("parameter_rate") for e in eff if isinstance(e.get("parameter_rate"), (int, float))]
        mems = [e.get("memory_mb") or e.get("gpu_memory_mb") for e in eff]
        mems = [m for m in mems if isinstance(m, (int, float)) and m]
        summary["parameter_rate"] = float(torch.tensor(rates, dtype=torch.float64).mean()) if rates else None
        summary["gpu_memory_mb"] = float(torch.tensor(mems, dtype=torch.float64).mean()) if mems else None
    return summary


def compare_to_paper(report: Dict[str, Any], backbone: str = "ddpm",
                     method: Optional[str] = None,
                     rel_tol: float = 0.25) -> Dict[str, Any]:
    """Diff measured numbers against the paper's Tables 1/2/4 (§5.2, §5.3, B.2)."""
    method = method or ("LDM-ANT" if str(backbone).lower() == "ldm" else "DDPM-ANT")
    task = report.get("task")
    task = str(task).lower() if task else None
    # normalise task keys such as "ffhq_sunglasses_ldm" -> "sunglasses"
    for name in ("sunglasses", "babies", "raphael", "sketches", "amedeo",
                 "haunted_houses", "haunted", "landscape_drawings", "landscape"):
        if task and name in task:
            task = "haunted_houses" if name == "haunted" else (
                "landscape_drawings" if name == "landscape" else name)
            break

    out: Dict[str, Any] = {"method": method, "task": task, "rel_tol": rel_tol, "notes": []}

    # Intra-LPIPS
    il = report.get("intra_lpips")
    measured_il = il.get("intra_lpips") if isinstance(il, dict) else il
    ref = PAPER_INTRA_LPIPS.get(method, {}).get(task) if task else None
    if measured_il is not None and ref is not None:
        target, std = ref
        diff = float(measured_il) - target
        out["intra_lpips"] = {
            "measured": float(measured_il),
            "paper": target,
            "paper_std": std,
            "diff": diff,
            "within_tol": abs(diff) <= max(rel_tol * abs(target), (std or 0.0)),
        }
    elif measured_il is not None:
        out["intra_lpips"] = {"measured": float(measured_il), "paper": None}
        out["notes"].append(f"no Intra-LPIPS reference for method={method}, task={task}")

    # FID
    fid = report.get("fid")
    if isinstance(fid, dict):
        out["fid"] = {}
        for target_name, res in fid.items():
            val = res.get("fid") if isinstance(res, dict) else res
            paper_val = PAPER_FID.get(method, {}).get(str(target_name).lower())
            if val is None or val != val or paper_val is None:
                out["fid"][target_name] = {"measured": val, "paper": paper_val}
                continue
            diff = float(val) - float(paper_val)
            out["fid"][target_name] = {
                "measured": float(val),
                "paper": float(paper_val),
                "diff": diff,
                "within_tol": abs(diff) <= rel_tol * float(paper_val),
            }

    # Efficiency
    eff = report.get("efficiency")
    rate = report.get("parameter_rate")
    if isinstance(eff, dict):
        rate = eff.get("parameter_rate", rate)
    if rate is not None and method in PAPER_PARAM_RATE:
        out["parameter_rate"] = {
            "measured": float(rate),
            "paper": PAPER_PARAM_RATE[method],
            "diff": float(rate) - PAPER_PARAM_RATE[method],
        }
    return out


def format_report(report: Dict[str, Any]) -> str:
    """Render a human-readable text summary of an evaluation report."""
    lines: List[str] = []
    lines.append("=" * 68)
    lines.append("DPMs-ANT evaluation report")
    lines.append("=" * 68)
    lines.append(f"method     : {report.get('method')}")
    lines.append(f"backbone   : {report.get('backbone')}")
    lines.append(f"task       : {report.get('task')}")
    if report.get("num_seeds"):
        lines.append(f"seeds      : {report.get('seeds')}  (n={report.get('num_seeds')})")
    if report.get("num_generated"):
        lines.append(f"generated  : {report.get('num_generated')} images "
                     f"in {report.get('generation_seconds', 0.0):.1f}s")

    il = report.get("intra_lpips")
    if isinstance(il, dict) and il.get("intra_lpips") is not None:
        std = report.get("intra_lpips_std")
        suffix = f" +/- {std:.4f}" if isinstance(std, (int, float)) else ""
        lines.append(f"Intra-LPIPS: {il['intra_lpips']:.4f}{suffix}  (higher is better)")
    elif isinstance(il, (int, float)):
        lines.append(f"Intra-LPIPS: {float(il):.4f}  (higher is better)")

    fid = report.get("fid")
    if isinstance(fid, dict) and fid:
        lines.append("FID (lower is better):")
        for target, res in fid.items():
            val = res.get("fid") if isinstance(res, dict) else res
            std = (report.get("fid_std") or {}).get(target)
            suffix = f" +/- {std:.4f}" if isinstance(std, (int, float)) else ""
            lines.append(f"  {target:<20s}: {val if val is None else f'{float(val):.4f}'}{suffix}")

    rate = report.get("parameter_rate")
    if isinstance(rate, (int, float)):
        lines.append(f"parameter rate: {100.0 * float(rate):.2f}%  "
                     f"(paper: {100.0 * PAPER_PARAM_RATE.get(report.get('method', ''), float('nan')):.2f}%)")
    mem = report.get("gpu_memory_mb")
    if isinstance(mem, (int, float)) and mem:
        lines.append(f"peak GPU memory: {float(mem):.0f} MB")

    comp = report.get("comparison")
    if isinstance(comp, dict):
        lines.append("-" * 68)
        lines.append("comparison against the paper")
        if isinstance(comp.get("intra_lpips"), dict) and "paper" in comp["intra_lpips"]:
            c = comp["intra_lpips"]
            if c.get("paper") is not None:
                lines.append(f"  Intra-LPIPS measured={c['measured']:.4f} paper={c['paper']:.4f} "
                             f"diff={c.get('diff', float('nan')):+.4f} "
                             f"ok={c.get('within_tol')}")
        for target, c in (comp.get("fid") or {}).items():
            if isinstance(c, dict) and c.get("paper") is not None:
                lines.append(f"  FID[{target}] measured={c['measured']:.4f} paper={c['paper']:.4f} "
                             f"diff={c.get('diff', float('nan')):+.4f} ok={c.get('within_tol')}")
            elif isinstance(c, dict):
                lines.append(f"  FID[{target}] measured={c.get('measured')} paper=<none>")
        for note in comp.get("notes", []):
            lines.append(f"  note: {note}")
    lines.append("=" * 68)
    return "\n".join(lines)


def save_report(report: Dict[str, Any], path: str, indent: int = 2) -> str:
    """Persist a report (JSON) and return the written path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=indent, default=str)
    LOGGER.info("wrote report to %s", path)
    return path


def load_report(path: str) -> Dict[str, Any]:
    """Load a previously saved JSON report."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------
def evaluate_model(model: torch.nn.Module,
                   classifier: Optional[torch.nn.Module] = None,
                   autoencoder: Optional[torch.nn.Module] = None,
                   cfg: Optional[Dict[str, Any]] = None,
                   backbone: str = "ddpm",
                   task: Optional[str] = None,
                   reference_dir: Optional[str] = None,
                   reference_images: Optional[torch.Tensor] = None,
                   num_samples: Optional[int] = None,
                   num_steps: Optional[int] = None,
                   seeds: Optional[Sequence[int]] = None,
                   device: Optional[str] = None,
                   out_dir: Optional[str] = None,
                   report_path: Optional[str] = None,
                   compute_intra_lpips: Optional[bool] = None,
                   compute_fid: Optional[bool] = None,
                   return_details: bool = False,
                   save_images: Optional[bool] = None,
                   verbose: bool = True,
                   **overrides: Any) -> Dict[str, Any]:
    """Evaluate one adapted model over one or more seeds and aggregate results."""
    eval_cfg = EvalConfig.from_dict(cfg, backbone=backbone, task=task, **overrides)
    if device is not None:
        eval_cfg.device = device
    if num_samples is not None:
        eval_cfg.num_samples = int(num_samples)
        eval_cfg.intra_lpips_num_samples = int(num_samples)
    if num_steps is not None:
        eval_cfg.num_steps = int(num_steps)
    if save_images is not None:
        eval_cfg.save_images = bool(save_images)
    eval_cfg.verbose = bool(verbose)
    if seeds:
        eval_cfg.seeds = [int(s) for s in seeds]

    evaluator = Evaluator(
        model=model,
        classifier=classifier,
        autoencoder=autoencoder,
        config=eval_cfg,
        cfg=cfg,
        backbone=backbone,
        task=task,
        device=eval_cfg.device,
        reference_dir=reference_dir,
        reference_images=reference_images,
    )

    reports: List[Dict[str, Any]] = []
    for seed in eval_cfg.seeds:
        sample_dir = None
        if out_dir is not None or eval_cfg.save_images:
            sample_dir = os.path.join(out_dir or (eval_cfg.sample_dir or "samples"),
                                      f"seed_{seed}")
        rep = evaluator.evaluate(num_samples=eval_cfg.num_samples,
                                 seed=seed,
                                 compute_intra_lpips=compute_intra_lpips,
                                 compute_fid=compute_fid,
                                 return_details=return_details,
                                 out_dir=sample_dir)
        reports.append(rep)
        if verbose:
            LOGGER.info("seed %d: Intra-LPIPS=%s FID=%s", seed,
                        rep.get("intra_lpips"), rep.get("fid"))

    summary = aggregate_seeds(reports)
    summary["per_seed"] = reports
    summary["method"] = "LDM-ANT" if str(backbone).lower() == "ldm" else "DDPM-ANT"
    summary["comparison"] = compare_to_paper(summary, backbone=backbone,
                                             rel_tol=eval_cfg.rel_tol)
    if verbose:
        print(format_report(summary))
    if report_path:
        save_report(summary, report_path)
    return summary


def evaluate_from_dirs(generated_dir: str,
                       reference_dir: Optional[str] = None,
                       fid_reference_dirs: Optional[Dict[str, str]] = None,
                       cfg: Optional[Dict[str, Any]] = None,
                       backbone: str = "ddpm",
                       task: Optional[str] = None,
                       num_samples: Optional[int] = None,
                       device: Optional[str] = None,
                       compute_intra_lpips: bool = True,
                       compute_fid: bool = True,
                       report_path: Optional[str] = None,
                       return_details: bool = False,
                       **overrides: Any) -> Dict[str, Any]:
    """Evaluate a directory of *already generated* images (baseline style).

    Useful for scoring the outputs of DDPM-PA / GAN baselines with exactly the
    same Intra-LPIPS and FID scripts (Tables 1-2, 4, 9).
    """
    eval_cfg = EvalConfig.from_dict(cfg, backbone=backbone, task=task, **overrides)
    if device is not None:
        eval_cfg.device = device
    n = int(num_samples or eval_cfg.num_samples)
    generated = load_images_from_dir(generated_dir, size=256, limit=n)
    LOGGER.info("loaded %d generated images from %s", generated.shape[0], generated_dir)

    report: Dict[str, Any] = {
        "method": "baseline",
        "backbone": backbone,
        "task": task,
        "generated_dir": generated_dir,
        "num_generated": int(generated.shape[0]),
    }

    if compute_intra_lpips:
        from .intra_lpips import compute_intra_lpips as _il  # local import

        ref_dir = reference_dir or resolve_reference_dir(task, cfg)
        report["intra_lpips"] = _il(
            generated,
            train_dir=ref_dir,
            num_samples=min(int(eval_cfg.intra_lpips_num_samples), int(generated.shape[0])),
            net=eval_cfg.lpips_net,
            device=eval_cfg.device,
            batch_size=max(1, int(eval_cfg.batch_size)),
            include_singletons=eval_cfg.include_singletons,
            cluster_average=eval_cfg.cluster_average,
            return_details=return_details,
        )

    if compute_fid:
        from .fid import compute_fid as _fid  # local import

        target_dirs = dict(fid_reference_dirs or {})
        targets = list(target_dirs) or list(eval_cfg.fid_targets)
        fid_out: Dict[str, Any] = {}
        for target in targets:
            ref = target_dirs.get(target) or _config_dir(cfg, target, [str(target).lower()])
            try:
                fid_out[target] = _fid(
                    generated,
                    reference_dir=ref,
                    cfg=cfg,
                    device=eval_cfg.device,
                    backend=eval_cfg.fid_backend,
                    batch_size=eval_cfg.fid_batch_size,
                    target=target,
                    return_details=return_details,
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("FID failed for %s: %s", target, exc)
                fid_out[target] = {"fid": float("nan"), "error": str(exc), "target": target}
        report["fid"] = fid_out

    report["comparison"] = compare_to_paper(report, backbone=backbone, rel_tol=eval_cfg.rel_tol)
    if report_path:
        save_report(report, report_path)
    return report


def run_evaluation(targets: Union[str, Sequence[str], Callable[[str], torch.nn.Module]],
                   cfg: Optional[Dict[str, Any]] = None,
                   backbone: str = "ddpm",
                   model_loader: Optional[Callable[[str, str], torch.nn.Module]] = None,
                   **kwargs: Any) -> Dict[str, Any]:
    """Evaluate one or several tasks and return ``{task: summary}``.

    ``targets`` may be a task name, a list of task names, or a callable
    ``task -> model`` (in which case ``backbone`` is used as given).
    """
    if isinstance(targets, str):
        targets = [targets]
    if not isinstance(targets, (list, tuple)):
        raise TypeError("targets must be a task name, a list of names, or a callable")

    results: Dict[str, Any] = {}
    for task in targets:
        task_backbone = backbone
        if isinstance(cfg, dict):
            task_cfg = (cfg.get("tasks") or {}).get(task, {})
            if isinstance(task_cfg, dict) and task_cfg.get("backbone"):
                task_backbone = task_cfg["backbone"]
        if callable(targets):
            model = targets(task)
        elif model_loader is not None:
            model = model_loader(task, task_backbone)
        else:
            raise ValueError("run_evaluation requires `model_loader` (or `targets` as a callable)")
        results[task] = evaluate_model(model, cfg=cfg, backbone=task_backbone,
                                       task=task, **kwargs)
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI glue
    """CLI: evaluate a directory of generated images, or a checkpoint's samples."""
    import argparse

    parser = argparse.ArgumentParser(description="DPMs-ANT evaluation (Intra-LPIPS / FID)")
    parser.add_argument("--generated-dir", type=str, default=None,
                        help="directory of generated images to score")
    parser.add_argument("--reference-dir", type=str, default=None,
                        help="10-shot training-image directory (Intra-LPIPS clustering)")
    parser.add_argument("--fid-ref", action="append", default=[],
                        metavar="TARGET=DIR", help="FID reference dir per target")
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config")
    parser.add_argument("--backbone", type=str, default="ddpm", choices=["ddpm", "ldm"])
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args(argv)

    cfg: Dict[str, Any] = {}
    if args.config:
        try:
            import yaml  # type: ignore

            with open(args.config, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception as exc:
            LOGGER.warning("could not load config %s: %s", args.config, exc)

    fid_dirs: Dict[str, str] = {}
    for item in args.fid_ref:
        if "=" in item:
            k, v = item.split("=", 1)
            fid_dirs[k.strip()] = v.strip()

    if args.generated_dir:
        report = evaluate_from_dirs(
            args.generated_dir,
            reference_dir=args.reference_dir,
            fid_reference_dirs=fid_dirs,
            cfg=cfg,
            backbone=args.backbone,
            task=args.task,
            num_samples=args.num_samples,
            device=args.device,
            report_path=args.report,
            return_details=args.details,
        )
    else:
        raise SystemExit("--generated-dir is required for the standalone evaluation CLI "
                         "(use scripts/sample.py + this CLI, or main.py --mode eval)")

    print(format_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
