"""Evaluation entry point for *Stochastic Interpolants with Data-Dependent Couplings*.

This script orchestrates the paper's evaluation protocol and reproduces its reported
numbers/figures:

* **Table 2 -- in-painting FID-50k** (Section 4.1): the dependent coupling (ours)
  reaches ``1.13`` against the uncoupled-interpolant baseline at ``1.35``.  The
  baseline is a Gaussian :math:`\\rho_0` with an *independent* coupling to
  :math:`\\rho_1`, i.e. :math:`x_0 = \\zeta` instead of
  :math:`x_0 = \\xi \\circ x_1 + (1-\\xi) \\circ \\zeta`.

* **Table 3 -- super-resolution FID-50k, 64x64 -> 256x256** (Section 4.2): the
  dependent coupling (ours) reaches ``2.13`` (train) / ``2.05`` (valid).  The
  baselines (Improved DDPM 12.26, SR3 11.30/5.20, ADM 7.49/3.10, Cascaded
  Diffusion 4.88/4.63, :math:`I^2`SB 2.70) are *reported* numbers taken from the
  cited works and are **not** re-run here.

* **Figures 3-6**: base/model/ground-truth triples (Figs. 3, 4, 6) and temporal
  probability-flow slices (Fig. 5).

* **Structural / theoretical validation**:
  - Proposition 3.1 transport-cost bound: ``E[|I_dot_t|^2]`` is smaller for the
    data-dependent coupling than for the independent base
    (for :math:`m(x_1)=x_1, C=\\sigma^2 I` the paper's analysis gives
    :math:`d\\sigma^2` versus :math:`2 E[|x_1|^2] + d\\sigma^2`).
  - In-painting invariant :math:`\\xi \\circ I_t = \\xi \\circ x_1` for every
    :math:`t` (Section 4.1): the velocity field vanishes outside the mask, the
    network output is masked accordingly, and the unmasked pixels are reproduced
    exactly by the ODE sample.

Usage
-----
::

    python evaluate.py --config configs/inpainting_256.yaml \
        --checkpoint runs/inpainting_256/checkpoints/step_200000.pt \
        --fid --qualitative --transport-cost --structural

    # quick, dependency-light smoke test (no ImageNet, no checkpoint)
    python evaluate.py --self-test

Nothing heavy is imported at module load time: ``torch``, ``si.*``, ``eval.*``,
``yaml`` and ``matplotlib`` are all resolved lazily so that
``--help``/``--self-test`` work in a bare environment.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Repo-root import shim so the file can be executed directly (python evaluate.py ...)
# --------------------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logger = logging.getLogger("evaluate")


# ======================================================================================
# Paper constants (Tables 2 and 3, Figures 3-6)
# ======================================================================================

#: FID-50k targets from Table 2 (in-painting on ImageNet) and Table 3 (super-resolution).
PAPER_FIDS: Dict[str, Dict[str, Optional[float]]] = {
    "inpainting": {"ours": 1.13, "baseline": 1.35},
    "superres_64_256": {"ours_valid": 2.05, "ours_train": 2.13},
    "superres_256_512": {"ours_valid": None, "ours_train": None},  # qualitative only (Fig. 6)
}

#: Baselines whose FIDs are *cited* from prior work (Table 3) and must not be re-run.
REPORTED_ONLY_BASELINES: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {
    # model -> (train FID, valid FID)
    "improved_ddpm": (12.26, None),
    "sr3": (11.30, 5.20),
    "adm": (7.49, 3.10),
    "cascaded_diffusion": (4.88, 4.63),
    "i2sb": (None, 2.70),
}

#: Table 2 baseline description (uncoupled interpolant: independent Gaussian base).
BASELINE_DESCRIPTION = "Uncoupled Interpolant (Baseline)"
OURS_DESCRIPTION = "Dependent Coupling (Ours)"

#: Solver names handled by the adaptive torchdiffeq integrators.
ADAPTIVE_METHODS = ("dopri5", "dopri8", "bosh3", "fehlberg2", "adaptive_heun")
FIXED_METHODS = ("euler", "midpoint", "rk4", "explicit_adams", "implicit_adams")


# ======================================================================================
# Small helpers
# ======================================================================================


def deep_update(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    out = copy.deepcopy(base)
    if not override:
        return out
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _require(module: str, attr: Optional[str] = None):
    """Lazily import ``module`` (optionally an attribute) with a friendly error."""
    try:
        mod = __import__(module, fromlist=["*"])
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(f"could not import '{module}': {exc}") from exc
    return getattr(mod, attr) if attr else mod


def _read_yaml(path: str) -> Dict[str, Any]:
    """Read a YAML config, falling back to the repo loader and then to JSON."""
    try:
        from si.utils.config import load_config  # type: ignore

        cfg = load_config(path)
        if isinstance(cfg, dict):
            return cfg
    except Exception:
        pass
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)


def make_generator(seed: Optional[int], device=None):
    """Build a ``torch.Generator`` when possible, else return ``None``."""
    if seed is None:
        return None
    try:
        import torch  # noqa: F401

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return generator
    except Exception:  # pragma: no cover - torch always present in practice
        return None


def _tolist(value, max_items: int = 8) -> List[Any]:
    """Best-effort conversion of a metric tensor/value into a short python list."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            flat = value.detach().reshape(-1).tolist()
            return flat[:max_items]
    except Exception:  # pragma: no cover
        pass
    if isinstance(value, (list, tuple)):
        return list(value)[:max_items]
    return [value]


# ======================================================================================
# Configuration
# ======================================================================================

DEFAULT_EVAL_CONFIG: Dict[str, Any] = {
    # --- task / data -----------------------------------------------------------------
    "name": "evaluate",
    "task": "inpainting",  # inpainting | superres
    "seed": 0,
    "dataset": "imagenet-1k",
    "split": "train",  # "train" or "validation" (Table 3 reports both)
    "resolution": 256,
    "low_resolution": None,
    "num_classes": 1000,
    "channels": 3,
    "data_range": "[-1, 1]",
    "cache_dir": None,
    "num_workers": 4,
    # --- coupling / interpolant ------------------------------------------------------
    "coupling": "inpainting",
    "base_mode": "coupled",  # "independent" reproduces the Table 2 baseline
    "coefficients": "inpainting",  # alpha_t=t, beta_t=1-t, gamma_t=0 (Sec. 4.1)
    "sigma": 1.0,
    "requires_conditioning": True,
    "num_tiles": 64,
    "missing_prob": 0.3,
    "expand_channels": False,
    "down_mode": "area",
    "up_mode": "bilinear",
    "antialias": True,
    # --- model (Appendix B) -----------------------------------------------------------
    "model": "unet",
    "in_channels": 3,
    "out_channels": 3,
    "image_size": 256,
    "dim_mults": (1, 1, 2, 3, 4),
    "resnet_block_groups": 8,
    "class_dropout_prob": 0.1,
    "learned_sinusoidal_cond": True,
    "learned_sinusoidal_dim": 32,
    "attention_dim_head": 64,
    "attention_heads": 4,
    "random_fourier_features": False,
    "conditioning_channels": 1,
    "mask_observed": True,
    # --- sampling (Algorithm 2) -------------------------------------------------------
    "method": "dopri5",
    "sampling_steps": 50,
    "atol": 1e-5,
    "rtol": 1e-5,
    "project_observed": True,
    "num_samples": 50000,
    "sample_batch_size": 32,
    # --- evaluation -------------------------------------------------------------------
    "eval_fid": True,
    "fid_num_samples": 50000,
    "fid_batch_size": 32,
    "fid_dataset": "imagenet",
    "qualitative": True,
    "qualitative_count": 8,
    "transport_cost": True,
    "transport_cost_samples": 20000,
    "structural": True,
    "output_dir": None,
}

#: CLI argument -> config key mapping.
_CLI_TO_CONFIG: Dict[str, str] = {
    "task": "task",
    "dataset": "dataset",
    "split": "split",
    "resolution": "resolution",
    "low_resolution": "low_resolution",
    "method": "method",
    "steps": "sampling_steps",
    "num_samples": "num_samples",
    "batch_size": "sample_batch_size",
    "coupling": "coupling",
    "coefficients": "coefficients",
    "sigma": "sigma",
    "base_mode": "base_mode",
    "output_dir": "output_dir",
    "seed": "seed",
}


def resolve_task_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the paper's per-task conventions (Sections 4.1 and 4.2).

    In-painting uses :math:`\\alpha_t=t, \\beta_t=1-t` and a single mask channel;
    super-resolution uses the deterministic (``gamma0``) preset with the upsampled
    low-resolution image appended to the channels of ``x``.
    """
    cfg = copy.deepcopy(cfg)
    task = str(cfg.get("task", "inpainting")).lower()
    if task in ("inpainting", "inpaint", "mask"):
        cfg["task"] = "inpainting"
        cfg.setdefault("coupling", "inpainting")
        cfg["coefficients"] = cfg.get("coefficients") or "inpainting"
        cfg["sigma"] = float(cfg.get("sigma", 1.0))
        cfg["conditioning_channels"] = 1
        cfg["mask_observed"] = True
        cfg["project_observed"] = True
        cfg["num_tiles"] = int(cfg.get("num_tiles", 64))
        cfg["missing_prob"] = float(cfg.get("missing_prob", 0.3))
        cfg.setdefault("fid_key", "inpainting")
        cfg.setdefault("paper_fid", PAPER_FIDS["inpainting"]["ours"])
        cfg.setdefault("paper_fid_baseline", PAPER_FIDS["inpainting"]["baseline"])
    elif task in ("superres", "superresolution", "sr"):
        cfg["task"] = "superres"
        cfg["coupling"] = "superres"
        cfg["coefficients"] = cfg.get("coefficients") or "gamma0"
        cfg["sigma"] = float(cfg.get("sigma", 0.05))
        low = cfg.get("low_resolution")
        if low is None:
            res = int(cfg.get("resolution", 256))
            low = 64 if res <= 256 else 256
        cfg["low_resolution"] = int(low)
        cfg["conditioning_channels"] = int(cfg.get("in_channels", 3))
        cfg["mask_observed"] = False
        cfg.setdefault("project_observed", False)
        cfg.setdefault("down_mode", "area")
        cfg.setdefault("up_mode", "bilinear")
        key = f"superres_{int(cfg['low_resolution'])}_{int(cfg.get('resolution', 256))}"
        cfg["fid_key"] = key
        targets = PAPER_FIDS.get(key, {})
        cfg.setdefault("paper_fid", targets.get("ours_valid"))
        cfg.setdefault("paper_fid_train", targets.get("ours_train"))
        cfg.setdefault("paper_fid_valid", targets.get("ours_valid"))
    else:
        raise ValueError(f"unknown task '{task}' (expected 'inpainting' or 'superres')")
    cfg.setdefault("image_size", int(cfg.get("resolution", 256)))
    return cfg


def build_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compose ``DEFAULT_EVAL_CONFIG`` <- YAML file <- ``overrides`` (+ task defaults)."""
    cfg = copy.deepcopy(DEFAULT_EVAL_CONFIG)
    if path:
        cfg = deep_update(cfg, _read_yaml(path))
    if overrides:
        clean = {k: v for k, v in overrides.items() if v is not None}
        cfg = deep_update(cfg, clean)
    cfg = resolve_task_settings(cfg)
    # keep model hyperparameters consistent with the requested resolution
    cfg["image_size"] = int(cfg.get("image_size") or cfg.get("resolution", 256))
    return cfg


# ======================================================================================
# Builder helpers (thin wrappers around the repo's public API)
# ======================================================================================


def build_coupling(cfg: Dict[str, Any]):
    """Instantiate the task coupling (Section 4.1 in-painting / Section 4.2 SR)."""
    task = cfg["task"]
    if task == "inpainting":
        try:
            from si.couplings import InpaintingCoupling  # type: ignore

            return InpaintingCoupling(
                sigma=float(cfg.get("sigma", 1.0)),
                num_tiles=int(cfg.get("num_tiles", 64)),
                missing_prob=float(cfg.get("missing_prob", 0.3)),
                expand_channels=bool(cfg.get("expand_channels", False)),
                coefficients=cfg.get("coefficients", "inpainting"),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("falling back to get_coupling('inpainting'): %s", exc)
            get_coupling = _require("si.couplings", "get_coupling")
            return get_coupling("inpainting", sigma=float(cfg.get("sigma", 1.0)))
    from si.couplings import SuperresCoupling  # type: ignore

    return SuperresCoupling(
        low_res=int(cfg.get("low_resolution", 64)),
        sigma=float(cfg.get("sigma", 0.05)),
        down_mode=cfg.get("down_mode", "area"),
        up_mode=cfg.get("up_mode", "bilinear"),
        antialias=bool(cfg.get("antialias", True)),
        coefficients=cfg.get("coefficients", "gamma0"),
    )


def build_interpolant(cfg: Dict[str, Any]):
    """Build the stochastic interpolant ``I_t`` for the configured preset."""
    Interpolant = _require("si.interpolants", "Interpolant")
    get_coefficients = _require("si.interpolants", "get_coefficients")
    return Interpolant(get_coefficients(cfg.get("coefficients", "inpainting")))


def build_model(cfg: Dict[str, Any], coupling=None):
    """Instantiate the Appendix-B velocity U-Net :math:`\\hat b_t(x, \\xi)`."""
    unet_from_config = _require("si.models", "unet_from_config")
    kwargs: Dict[str, Any] = {
        "in_channels": int(cfg.get("in_channels", 3)),
        "out_channels": int(cfg.get("out_channels", cfg.get("in_channels", 3))),
        "channels": int(cfg.get("channels", 256)),
        "dim_mults": tuple(cfg.get("dim_mults", (1, 1, 2, 3, 4))),
        "resnet_block_groups": int(cfg.get("resnet_block_groups", 8)),
        "num_classes": int(cfg.get("num_classes", 1000)),
        "class_dropout_prob": float(cfg.get("class_dropout_prob", 0.1)),
        "learned_sinusoidal_cond": bool(cfg.get("learned_sinusoidal_cond", True)),
        "learned_sinusoidal_dim": int(cfg.get("learned_sinusoidal_dim", 32)),
        "attention_dim_head": int(cfg.get("attention_dim_head", 64)),
        "attention_heads": int(cfg.get("attention_heads", 4)),
        "random_fourier_features": bool(cfg.get("random_fourier_features", False)),
        "conditioning_channels": int(cfg.get("conditioning_channels", 0)),
        "mask_observed": bool(cfg.get("mask_observed", False)),
        "image_size": int(cfg.get("image_size", 256)),
    }
    return unet_from_config(**kwargs)


def load_checkpoint(model, path: Optional[str], device=None, use_ema: bool = True) -> Dict[str, Any]:
    """Load weights from a training checkpoint (prefers EMA weights when present)."""
    if not path:
        logger.warning("no checkpoint given -- evaluating a randomly initialised model")
        return {}
    try:
        from train import Trainer  # type: ignore

        return Trainer.load_into(model, path, map_location=device, use_ema=use_ema)
    except Exception:
        import torch

        ckpt = torch.load(path, map_location=device or "cpu")
        if isinstance(ckpt, dict):
            for key in ("ema", "model", "state_dict", "model_state_dict"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    prefix = "module." if any(k.startswith("module.") for k in ckpt[key]) else ""
                    model.load_state_dict({k.replace(prefix, ""): v for k, v in ckpt[key].items()}, strict=False)
                    break
        return ckpt if isinstance(ckpt, dict) else {}


# ======================================================================================
# Sampling / generation
# ======================================================================================


def probability_flow_ode(*args, **kwargs):
    """Lazy access to the probability-flow ODE sampler (Section 3.4 / Eq. 8)."""
    return _require("si.samplers", "probability_flow_ode")(*args, **kwargs)


def unpack_batch(batch) -> Tuple[Any, Optional[Any], Optional[Any]]:
    """Extract ``(x1, labels, low_res)`` from a dict- or tuple-style batch."""
    if isinstance(batch, dict):
        x1 = batch.get("x1", batch.get("image", batch.get("images")))
        labels = batch.get("label", batch.get("y", batch.get("labels")))
        low = batch.get("low", batch.get("low_res", batch.get("conditioning")))
        return x1, labels, low
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            return batch[0], batch[1], None
        return batch[0], batch[1], batch[2]
    return batch, None, None


def draw_base(coupling, cfg: Dict[str, Any], x1, xi=None, generator=None):
    """Build the base sample ``x_0`` drawn from the coupled (or independent) density.

    ``base_mode="coupled"`` yields the paper's data-dependent coupling:

    * in-painting: :math:`x_0 = \\xi \\circ x_1 + (1-\\xi) \\circ \\zeta`
    * super-resolution: :math:`x_0 = \\mathcal U(\\mathcal D(x_1)) + \\sigma \\zeta`

    ``base_mode="independent"`` yields :math:`x_0 = \\zeta`, reproducing the
    ``1.35`` uncoupled-interpolant baseline of Table 2.
    """
    import torch

    if str(cfg.get("base_mode", "coupled")).lower() in ("independent", "gaussian", "baseline", "uncoupled"):
        zeta = torch.randn(x1.shape, generator=generator, device=x1.device, dtype=x1.dtype)
        if xi is None and hasattr(coupling, "sample_xi"):
            try:
                xi = coupling.sample_xi(x1, generator=generator)
            except TypeError:
                xi = coupling.sample_xi(x1)
        return zeta, xi, zeta

    zeta = torch.randn(x1.shape, generator=generator, device=x1.device, dtype=x1.dtype)
    out = coupling.build_x0(x1, xi=xi, zeta=zeta, generator=generator, return_xi=True)
    if isinstance(out, tuple):
        x0, xi_out = out[0], out[1]
    else:  # pragma: no cover - defensive
        x0, xi_out = out, xi
    return x0, xi_out, zeta


def _ode_kwargs(cfg: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Solver-specific keyword arguments (adaptive vs fixed-step)."""
    if method in ADAPTIVE_METHODS:
        return {"atol": float(cfg.get("atol", 1e-5)), "rtol": float(cfg.get("rtol", 1e-5))}
    return {"steps": int(cfg.get("sampling_steps", 50))}


def sample_batch(model, coupling, cfg: Dict[str, Any], x1, low=None, labels=None,
                 generator=None, return_trajectory: bool = False) -> Dict[str, Any]:
    """Integrate the probability-flow ODE from ``t=0`` to ``t=1`` for one batch."""
    import torch

    device = x1.device
    task = cfg["task"]
    xi = None
    if task == "inpainting":
        try:
            xi = coupling.sample_xi(x1, generator=generator)
        except TypeError:
            xi = coupling.sample_xi(x1)
    else:
        if low is not None:
            xi = coupling.U(low, high_res=x1.shape[-2:])
        else:
            xi = coupling.UD(x1)

    x0, xi, zeta = draw_base(coupling, cfg, x1, xi=xi, generator=generator)

    method = str(cfg.get("method", "dopri5"))
    kwargs: Dict[str, Any] = {"method": method, "xi": xi, "y": labels}
    kwargs.update(_ode_kwargs(cfg, method))

    mask_fn = getattr(coupling, "mask_velocity", None)
    project = None
    if bool(cfg.get("project_observed", False)) and xi is not None and task == "inpainting":
        make_project_fn = _require("si.samplers", "make_project_fn")
        try:
            project = make_project_fn(xi=xi, x1=x1)
        except TypeError:  # pragma: no cover
            project = make_project_fn(xi, x1)

    with torch.no_grad():
        result = probability_flow_ode(
            model, x0, mask_fn=mask_fn, project=project,
            return_trajectory=return_trajectory, **kwargs,
        )
    if isinstance(result, tuple):
        sample, trajectory = result
    else:
        sample, trajectory = result, None

    out: Dict[str, Any] = {"sample": sample, "x0": x0, "x1": x1, "xi": xi, "zeta": zeta}
    if labels is not None:
        out["label"] = labels
    if low is not None:
        out["low"] = low
    if trajectory is not None:
        out["trajectory"] = trajectory
    return out


def build_loader(cfg: Dict[str, Any], num_samples: int, batch_size: int, fabric=None,
                 split: Optional[str] = None, synthetic: bool = False):
    """Build an ImageNet loader sized for ``num_samples`` (synthetic fallback allowed)."""
    build_dataloader = _require("si.data", "build_dataloader")
    split = split or cfg.get("split", "train")
    loader = build_dataloader(
        split=split,
        batch_size=batch_size,
        resolution=int(cfg.get("resolution", 256)),
        low_resolution=cfg.get("low_resolution"),
        num_workers=int(cfg.get("num_workers", 4)),
        synthetic=synthetic,
        max_samples=num_samples,
        cache_dir=cfg.get("cache_dir"),
        return_dict=True,
    )
    if fabric is not None:
        try:
            loader = fabric.setup_dataloaders(loader)
        except Exception:  # pragma: no cover - single process
            pass
    return loader


def generate_samples(model, coupling, cfg: Dict[str, Any], num_samples: Optional[int] = None,
                     loader=None, fabric=None, synthetic: bool = False,
                     qualitative_count: int = 8, progress: bool = True,
                     return_trajectory: bool = False) -> Dict[str, Any]:
    """Generate samples for FID-50k plus qualitative triples (Figs. 3/4/6)."""
    import torch

    num_samples = int(num_samples or cfg.get("num_samples", 50000))
    batch_size = int(cfg.get("sample_batch_size", 32))
    device = getattr(fabric, "device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    if loader is None:
        loader = build_loader(cfg, num_samples, batch_size, fabric=fabric,
                              synthetic=synthetic)

    generator = make_generator(cfg.get("seed"), device)

    samples: List[Any] = []
    bases: List[Any] = []
    truths: List[Any] = []
    masks: List[Any] = []
    labels: List[Any] = []
    lows: List[Any] = []
    trajectories: List[Any] = []
    qualitative: Dict[str, Any] = {}
    generated = 0
    start = time.time()

    iterator = loader
    if progress:
        try:
            from tqdm import tqdm  # type: ignore

            iterator = tqdm(loader, desc="sampling", leave=False)
        except Exception:  # pragma: no cover
            iterator = loader

    for batch in iterator:
        x1, y, low = unpack_batch(batch)
        if x1 is None:
            continue
        x1 = x1.to(device)
        if y is not None:
            y = y.to(device)
        if low is not None:
            low = low.to(device)
        remaining = num_samples - generated
        if remaining <= 0:
            break
        if x1.shape[0] > remaining:
            x1 = x1[:remaining]
            if y is not None:
                y = y[:remaining]
            if low is not None:
                low = low[:remaining]

        out = sample_batch(model, coupling, cfg, x1, low=low, labels=y,
                           generator=generator, return_trajectory=return_trajectory)
        samples.append(out["sample"].detach().float().cpu())
        bases.append(out["x0"].detach().float().cpu())
        truths.append(out["x1"].detach().float().cpu())
        if out.get("xi") is not None:
            masks.append(out["xi"].detach().float().cpu())
        if out.get("label") is not None:
            labels.append(out["label"].detach().cpu())
        if out.get("low") is not None:
            lows.append(out["low"].detach().float().cpu())
        if return_trajectory and out.get("trajectory") is not None:
            trajectories.append(out["trajectory"].detach().float().cpu())

        if len(qualitative.get("model", [])) < qualitative_count:
            take = min(qualitative_count - len(qualitative.get("model", [])), out["sample"].shape[0])
            qualitative.setdefault("base", []).append(out["x0"][:take].detach().float().cpu())
            qualitative.setdefault("model", []).append(out["sample"][:take].detach().float().cpu())
            qualitative.setdefault("ground_truth", []).append(out["x1"][:take].detach().float().cpu())
            if out.get("xi") is not None:
                qualitative.setdefault("xi", []).append(out["xi"][:take].detach().float().cpu())
            if out.get("label") is not None:
                qualitative.setdefault("label", []).append(out["label"][:take].detach().cpu())
            if return_trajectory and out.get("trajectory") is not None:
                qualitative.setdefault("trajectory", []).append(
                    out["trajectory"][:, :take].detach().float().cpu()
                )
        generated += int(out["sample"].shape[0])

    def _cat(chunks: List[Any]):
        if not chunks:
            return None
        return torch.cat(chunks, dim=0)

    payload: Dict[str, Any] = {
        "samples": _cat(samples),
        "base": _cat(bases),
        "ground_truth": _cat(truths),
        "xi": _cat(masks),
        "labels": _cat(labels),
        "low": _cat(lows),
        "trajectory": _cat(trajectories) if trajectories else None,
        "num_generated": generated,
        "elapsed_sec": time.time() - start,
        "task": cfg["task"],
        "config": {k: v for k, v in cfg.items() if not isinstance(v, (dict, list, tuple))},
    }
    qualitative_flat: Dict[str, Any] = {}
    for key, chunks in qualitative.items():
        if key == "trajectory":
            try:
                qualitative_flat[key] = torch.cat(chunks, dim=1)  # (T, N, ...)
            except Exception:  # pragma: no cover
                qualitative_flat[key] = chunks[0]
        else:
            qualitative_flat[key] = _cat(chunks)
    payload["qualitative"] = qualitative_flat
    logger.info("generated %d samples in %.1fs", generated, payload["elapsed_sec"])
    return payload


# ======================================================================================
# Metrics: FID-50k (Tables 2 and 3)
# ======================================================================================


def compute_fid(payload: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compute FID-50k against standard ImageNet Inception statistics.

    Delegates to :mod:`eval.fid` (which implements the Frechet distance between
    Inception feature Gaussians).  Returns ``None`` if the evaluation stack is
    unavailable so the rest of the report still runs.
    """
    samples = payload.get("samples")
    if samples is None:
        return None
    try:
        evaluate_fid_50k = _require("eval.fid", "evaluate_fid_50k")
    except ImportError as exc:
        logger.warning("FID evaluation unavailable: %s", exc)
        return None

    key = cfg.get("fid_key") or cfg["task"]
    paper = cfg.get("paper_fid")
    try:
        result = evaluate_fid_50k(
            samples,
            task=key,
            batch_size=int(cfg.get("fid_batch_size", cfg.get("sample_batch_size", 32))),
            num_samples=int(cfg.get("fid_num_samples", 50000)),
            dataset=cfg.get("fid_dataset", "imagenet"),
            resolution=int(cfg.get("resolution", 256)),
            paper_fid=paper,
            verbose=True,
        )
    except Exception as exc:  # pragma: no cover - heavy dependency
        logger.warning("FID computation failed: %s", exc)
        return None

    if hasattr(result, "as_dict"):
        info = result.as_dict()
    elif isinstance(result, dict):
        info = dict(result)
    else:  # pragma: no cover
        info = {"fid": float(result)}
    info.setdefault("task", key)
    info["paper_fid"] = paper
    info["description"] = OURS_DESCRIPTION
    return info


# ======================================================================================
# Structural validation (Section 4.1)
# ======================================================================================


def structural_checks(payload: Dict[str, Any], cfg: Dict[str, Any], coupling=None) -> Dict[str, Any]:
    """Verify the in-painting invariant :math:`\\xi \\circ I_t = \\xi \\circ x_1`.

    Also confirms that the network velocity is structurally masked to zero on the
    observed pixels (Section 4.1: ``b_t(x, xi) = 0`` except in the masked regions),
    and reports the mean missing fraction of the masks used.
    """
    import torch

    report: Dict[str, Any] = {"task": cfg["task"]}
    if cfg["task"] != "inpainting":
        return report

    xi = payload.get("xi")
    x1 = payload.get("ground_truth")
    sample = payload.get("samples")
    if xi is None or x1 is None or sample is None:
        report["available"] = False
        return report

    if xi.shape[1] == 1 and x1.shape[1] > 1:
        xi_full = xi.expand(-1, x1.shape[1], -1, -1)
    else:
        xi_full = xi

    observed = xi_full > 0.5
    diff = (sample - x1).abs()
    max_err = float(diff[observed].max()) if observed.any() else 0.0
    mean_err = float(diff[observed].mean()) if observed.any() else 0.0
    report.update(
        {
            "available": True,
            "num_samples": int(sample.shape[0]),
            "observed_max_abs_error": max_err,
            "observed_mean_abs_error": mean_err,
            "unmasked_reproduced_exactly": max_err <= 1e-4,
            "missing_fraction_mean": float((~observed).float().mean()),
        }
    )

    # Verify the velocity output vanishes on the observed pixels.
    model = payload.get("_model")
    if model is not None:
        try:
            t = torch.rand(sample.shape[0], device=sample.device)
            velocity = model(sample, t, xi=payload["xi"].to(sample.device),
                             y=payload.get("labels").to(sample.device) if payload.get("labels") is not None else None)
            if isinstance(velocity, dict):
                velocity = velocity.get("velocity", velocity.get("b_hat", velocity))
            if isinstance(velocity, (tuple, list)):
                velocity = velocity[0]
            obs = observed.to(velocity.device)
            report["velocity_observed_max_abs"] = float(velocity.abs()[obs].max()) if obs.any() else 0.0
            report["velocity_masked_to_zero"] = report["velocity_observed_max_abs"] <= 1e-6
        except Exception as exc:  # pragma: no cover
            report["velocity_check_error"] = str(exc)
    return report


# ======================================================================================
# Proposition 3.1: transport-cost bound
# ======================================================================================


def transport_cost_comparison(cfg: Dict[str, Any], coupling=None, coefficients=None,
                              num_samples: Optional[int] = None, batch_size: int = 32,
                              synthetic: bool = True) -> Dict[str, Any]:
    """Empirically compare :math:`E[|I_dot_t|^2]` for coupled vs independent bases.

    Proposition 3.1 bounds the transport cost by :math:`\\int_0^1 E[|I_dot_t|^2] dt`.
    For the toy case :math:`m(x_1) = x_1`, :math:`C = \\sigma^2 I` the paper's
    expansion gives :math:`d\\sigma^2` for the coupled base versus
    :math:`2E[|x_1|^2] + d\\sigma^2` for the independent base, so the coupled value
    must be smaller.  This routine estimates both and reports the difference.
    """
    import torch

    try:
        transport_cost_estimate = _require("si.losses", "transport_cost_estimate")
        Interpolant = _require("si.interpolants", "Interpolant")
        get_coefficients = _require("si.interpolants", "get_coefficients")
    except ImportError as exc:  # pragma: no cover
        return {"available": False, "error": str(exc)}

    if coupling is None:
        coupling = build_coupling(cfg)
    name = cfg.get("coefficients", "inpainting")
    interp = Interpolant(get_coefficients(name))

    num_samples = int(num_samples or cfg.get("transport_cost_samples", 20000))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = make_generator(cfg.get("seed"))

    build_dataloader = None
    loader = None
    try:
        build_dataloader = _require("si.data", "build_dataloader")
        loader = build_dataloader(
            split=cfg.get("split", "train"),
            batch_size=batch_size,
            resolution=int(cfg.get("resolution", 256)),
            low_resolution=cfg.get("low_resolution"),
            num_workers=0,
            synthetic=synthetic,
            max_samples=num_samples,
            return_dict=True,
        )
    except Exception as exc:  # pragma: no cover - data unavailable
        logger.warning("transport-cost comparison using synthetic Gaussian targets: %s", exc)
        loader = None

    coupled_sum, independent_sum, count = 0.0, 0.0, 0
    toy_coupled, toy_independent = [], []
    while count < num_samples:
        if loader is not None:
            x1, _, low = unpack_batch(next(iter(loader)))
            x1 = x1.to(device)
        else:
            # Toy Gaussian target (matches the analytic Prop. 3.1 comparison).
            n = min(batch_size, num_samples - count)
            x1 = torch.randn(n, int(cfg.get("in_channels", 3)),
                             int(cfg.get("resolution", 256)), int(cfg.get("resolution", 256)),
                             generator=generator)
            x1 = x1.to(device)
            if x1.shape[0] == 0:
                break
        if x1.shape[0] == 0:
            break
        if count + x1.shape[0] > num_samples:
            x1 = x1[: num_samples - count]

        zeta = torch.randn(x1.shape, generator=generator, device=x1.device)
        # coupled base
        out = coupling.build_x0(x1, zeta=zeta, generator=generator, return_xi=True)
        x0_coupled = out[0] if isinstance(out, tuple) else out
        # independent base (Table 2 baseline)
        x0_independent = zeta

        for x0, acc in ((x0_coupled, "coupled"), (x0_independent, "independent")):
            try:
                cost = transport_cost_estimate(x0, x1, interpolant=interp, samples=8)
            except TypeError:
                cost = transport_cost_estimate(x0, x1, coefficients=interp, samples=8)
            value = float(cost)
            if acc == "coupled":
                coupled_sum += value * x1.shape[0]
                toy_coupled.append(value)
            else:
                independent_sum += value * x1.shape[0]
                toy_independent.append(value)
        count += int(x1.shape[0])

    n = max(count, 1)
    coupled = coupled_sum / n
    independent = independent_sum / n
    report = {
        "available": True,
        "coefficients": name,
        "num_samples": int(count),
        "transport_cost_coupled": coupled,
        "transport_cost_independent": independent,
        "difference": independent - coupled,
        "coupled_is_smaller": coupled < independent,
    }
    # Analytic toy reference (m(x1)=x1, C=sigma^2 I): d sigma^2 vs 2 E|x1|^2 + d sigma^2
    d = math.prod(cfg.get("in_channels", 3) for _ in range(1)) * 1
    d = int(cfg.get("in_channels", 3)) * int(cfg.get("resolution", 256)) ** 2
    sigma = float(cfg.get("sigma", 1.0))
    report["analytic_coupled_d_sigma_sq"] = d * sigma ** 2
    report["analytic_independent_extra"] = 2.0 * d  # E|x1|^2 = d for unit-variance x1
    return report


# ======================================================================================
# Qualitative figures (Figures 3-6)
# ======================================================================================


def save_qualitative(payload: Dict[str, Any], cfg: Dict[str, Any], outdir: str) -> List[str]:
    """Write base/model/GT triple grids (Figs. 3/4/6) and probability-flow slices (Fig. 5)."""
    os.makedirs(outdir, exist_ok=True)
    written: List[str] = []
    qualitative = payload.get("qualitative") or {}
    if not qualitative:
        logger.warning("no qualitative tensors available -- skipping figures")
        return written

    try:
        make_triples = _require("eval.qualitative", "make_triples")
        save_triples_grid = _require("eval.qualitative", "save_triples_grid")
        make_probability_flow_figure = _require("eval.qualitative", "make_probability_flow_figure")
    except ImportError as exc:  # pragma: no cover
        logger.warning("qualitative module unavailable: %s", exc)
        return written

    base = qualitative.get("base")
    model = qualitative.get("model")
    truth = qualitative.get("ground_truth")
    xi = qualitative.get("xi")
    labels = qualitative.get("label")

    if base is not None and model is not None and truth is not None:
        count = int(cfg.get("qualitative_count", 8))
        base, model, truth = base[:count], model[:count], truth[:count]
        labels_arg = labels[:count].tolist() if labels is not None else None
        xi_arg = xi[:count] if xi is not None else None
        triples = make_triples(base, model, truth, xi=xi_arg, class_labels=labels_arg)
        path = os.path.join(outdir, f"figure_triples_{cfg['task']}_{cfg.get('resolution', 256)}.png")
        written.append(save_triples_grid(triples, path))
        logger.info("wrote qualitative triples -> %s", path)

    trajectory = qualitative.get("trajectory")
    if trajectory is not None and trajectory.shape[0] > 1:
        try:
            path = os.path.join(outdir, f"figure_probability_flow_{cfg['task']}.png")
            result = make_probability_flow_figure(
                trajectory[:, 0] if trajectory.dim() > 4 else trajectory[0],
                ground_truth=(truth[0] if truth is not None else None),
                mask=(xi[0] if xi is not None else None),
                path=path,
            )
            written.append(result if isinstance(result, str) else path)
            logger.info("wrote probability-flow figure -> %s", path)
        except Exception as exc:  # pragma: no cover
            logger.warning("probability-flow figure failed: %s", exc)
    return written


# ======================================================================================
# Sample / payload persistence
# ======================================================================================


def save_samples(payload: Dict[str, Any], outdir: str, stem: str = "samples") -> str:
    """Persist the generated tensors (excluding the qualitative bundle) to ``.pt``."""
    import torch

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{stem}.pt")
    blob = {k: v for k, v in payload.items() if k not in ("qualitative", "_model", "config")}
    torch.save(blob, path)
    logger.info("saved samples -> %s", path)
    return path


# ======================================================================================
# Report assembly (Tables 2 and 3)
# ======================================================================================


def build_report(cfg: Dict[str, Any], fid_info: Optional[Dict[str, Any]],
                 structural: Optional[Dict[str, Any]],
                 transport: Optional[Dict[str, Any]],
                 figures: Optional[Sequence[str]] = None,
                 num_generated: Optional[int] = None) -> Dict[str, Any]:
    """Assemble the metric report that mirrors Tables 2 and 3."""
    task = cfg["task"]
    key = cfg.get("fid_key") or task
    report: Dict[str, Any] = {
        "task": task,
        "fid_key": key,
        "dataset": cfg.get("dataset", "imagenet-1k"),
        "split": cfg.get("split", "train"),
        "resolution": int(cfg.get("resolution", 256)),
        "low_resolution": cfg.get("low_resolution"),
        "num_samples": num_generated,
        "coefficients": cfg.get("coefficients"),
        "coupling": cfg.get("coupling"),
        "base_mode": cfg.get("base_mode", "coupled"),
        "sigma": cfg.get("sigma"),
        "method": cfg.get("method"),
        "sampling_steps": cfg.get("sampling_steps"),
        "fid": fid_info,
        "structural": structural,
        "transport_cost": transport,
        "figures": list(figures or []),
    }

    # -------- Table 2: in-painting ----------------------------------------------------
    if task == "inpainting":
        ours = (fid_info or {}).get("fid")
        report["table2"] = {
            "caption": "FID for Inpainting Task (ImageNet).",
            "rows": [
                {
                    "model": BASELINE_DESCRIPTION,
                    "fid_50k": PAPER_FIDS["inpainting"]["baseline"],
                    "source": "reported",
                },
                {
                    "model": OURS_DESCRIPTION,
                    "fid_50k": PAPER_FIDS["inpainting"]["ours"],
                    "source": "reported",
                },
            ],
            "measured": {"model": OURS_DESCRIPTION, "fid_50k": ours},
            "baseline_note": (
                "Reproduce the 1.35 uncoupled baseline with --base-mode independent "
                "(x_0 ~ N(0, I) instead of x_0 = xi o x_1 + (1-xi) o zeta)."
            ),
        }
        if ours is not None:
            report["table2"]["delta_to_paper"] = abs(float(ours) - float(PAPER_FIDS["inpainting"]["ours"]))
    # -------- Table 3: super-resolution ----------------------------------------------
    else:
        rows = [
            {"model": pretty, "train": train, "valid": valid, "source": "reported"}
            for pretty, (train, valid) in _baseline_rows().items()
        ]
        rows.append(
            {
                "model": OURS_DESCRIPTION,
                "train": PAPER_FIDS["superres_64_256"]["ours_train"],
                "valid": PAPER_FIDS["superres_64_256"]["ours_valid"],
                "source": "reported",
            }
        )
        report["table3"] = {
            "caption": "FID-50k for Super-resolution, 64x64 to 256x256.",
            "rows": rows,
            "measured": {"model": OURS_DESCRIPTION, "fid_50k": (fid_info or {}).get("fid")},
            "reported_only_baselines": sorted(REPORTED_ONLY_BASELINES),
            "note": "Baseline FIDs are cited from prior work and must not be re-run.",
        }
    return report


def _baseline_rows() -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Human-readable baseline names for Table 3."""
    pretty = {
        "improved_ddpm": "Improved DDPM",
        "sr3": "SR3",
        "adm": "ADM",
        "cascaded_diffusion": "Cascaded Diffusion",
        "i2sb": "I^2SB",
    }
    return {pretty[k]: v for k, v in REPORTED_ONLY_BASELINES.items() if k in pretty}


def format_report(report: Dict[str, Any]) -> str:
    """Render the report as human-readable text (metric tables + checks)."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("Stochastic Interpolants with Data-Dependent Couplings -- evaluation")
    lines.append("=" * 78)
    lines.append(f"task           : {report.get('task')}")
    lines.append(f"dataset/split  : {report.get('dataset')} / {report.get('split')}")
    lines.append(f"resolution     : {report.get('resolution')} (low-res {report.get('low_resolution')})")
    lines.append(f"sampler        : {report.get('method')} ({report.get('sampling_steps')} steps)")
    lines.append(f"coupling       : {report.get('coupling')} | coefficients {report.get('coefficients')}")
    lines.append(f"base_mode      : {report.get('base_mode')} | sigma {report.get('sigma')}")
    lines.append(f"samples        : {report.get('num_samples')}")
    lines.append("-" * 78)

    fid = report.get("fid") or {}
    if fid:
        lines.append(f"FID-50k (measured): {fid.get('fid')}")
        if fid.get("paper_fid") is not None:
            delta = abs(float(fid["fid"]) - float(fid["paper_fid"]))
            lines.append(f"FID-50k (paper)   : {fid['paper_fid']}  |delta| = {delta:.4f}")
    else:
        lines.append("FID-50k (measured): not computed")

    table2 = report.get("table2")
    if table2:
        lines.append("-" * 78)
        lines.append("Table 2 -- " + table2["caption"])
        lines.append(f"{'Model':<34}{'FID-50k':>10}")
        for row in table2["rows"]:
            lines.append(f"{row['model']:<34}{row['fid_50k']:>10}")
        if table2.get("delta_to_paper") is not None:
            lines.append(f"delta to paper: {table2['delta_to_paper']:.4f}")
        lines.append(table2["baseline_note"])

    table3 = report.get("table3")
    if table3:
        lines.append("-" * 78)
        lines.append("Table 3 -- " + table3["caption"])
        lines.append(f"{'Model':<26}{'Train':>10}{'Valid':>10}")
        for row in table3["rows"]:
            train = row["train"] if row["train"] is not None else "-"
            valid = row["valid"] if row["valid"] is not None else "-"
            lines.append(f"{row['model']:<26}{train!s:>10}{valid!s:>10}")
        lines.append("Baselines are reported values from prior work (not re-run).")

    structural = report.get("structural")
    if structural and structural.get("available"):
        lines.append("-" * 78)
        lines.append("Section 4.1 structural checks (xi o X_{t=1} = xi o x_1)")
        lines.append(f"observed max |error| : {structural['observed_max_abs_error']:.3e}")
        lines.append(f"unmasked reproduced  : {structural['unmasked_reproduced_exactly']}")
        lines.append(f"missing fraction     : {structural['missing_fraction_mean']:.4f} (paper p=0.3)")
        if "velocity_masked_to_zero" in structural:
            lines.append(f"velocity masked zero : {structural['velocity_masked_to_zero']} "
                         f"(max |b| = {structural['velocity_observed_max_abs']:.3e})")

    transport = report.get("transport_cost")
    if transport and transport.get("available"):
        lines.append("-" * 78)
        lines.append("Proposition 3.1 transport-cost comparison")
        lines.append(f"E[|I_dot|^2] coupled     : {transport['transport_cost_coupled']:.4f}")
        lines.append(f"E[|I_dot|^2] independent : {transport['transport_cost_independent']:.4f}")
        lines.append(f"coupled is smaller       : {transport['coupled_is_smaller']}")

    if report.get("figures"):
        lines.append("-" * 78)
        lines.append("figures:")
        for path in report["figures"]:
            lines.append(f"  {path}")
    lines.append("=" * 78)
    return "\n".join(lines)


def write_report(report: Dict[str, Any], outdir: str, stem: str = "evaluation_report") -> Tuple[str, str]:
    """Write JSON + text reports, returning their paths."""
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, f"{stem}.json")
    txt_path = os.path.join(outdir, f"{stem}.txt")

    def _default(obj):
        try:
            import torch

            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
        except Exception:  # pragma: no cover
            pass
        return str(obj)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=_default)
    text = format_report(report)
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    logger.info("wrote report -> %s / %s", json_path, txt_path)
    return json_path, txt_path


# ======================================================================================
# CLI
# ======================================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained stochastic-interpolant coupling model "
                    "(FID-50k, qualitative figures, Prop. 3.1 and Sec. 4.1 checks).",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config (configs/*.yaml)")
    parser.add_argument("--checkpoint", type=str, default=None, help="trained checkpoint (.pt)")
    parser.add_argument("--task", type=str, default=None, choices=["inpainting", "superres"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, help="train | validation")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--low-resolution", dest="low_resolution", type=int, default=None)
    parser.add_argument("--method", type=str, default=None, help="dopri5 | euler | rk4 | ...")
    parser.add_argument("--steps", type=int, default=None, help="Euler/N steps (Algorithm 2)")
    parser.add_argument("--sampling-steps", dest="steps_alias", type=int, default=None)
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--fid-batch-size", dest="fid_batch_size", type=int, default=None)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--coeffs", dest="coefficients", type=str, default=None)
    parser.add_argument("--base-mode", dest="base_mode", type=str, default=None,
                        help="coupled (ours) | independent (Table 2 baseline)")
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--reference-stats", dest="reference_stats", type=str, default=None,
                        help="precomputed FID reference statistics (.npz/.pt)")
    parser.add_argument("--fid-weights", dest="fid_weights", type=str, default=None,
                        help="pytorch-fid Inception weights for exact literature FID")
    parser.add_argument("--samples", type=str, default=None,
                        help="evaluate an existing payload .pt instead of generating samples")
    parser.add_argument("--synthetic", action="store_true",
                        help="use random tensors instead of downloading ImageNet")
    parser.add_argument("--fid", action="store_true", help="compute FID-50k")
    parser.add_argument("--no-fid", dest="fid_off", action="store_true", help="skip FID")
    parser.add_argument("--qualitative", action="store_true", help="write Figs. 3/4/6 (+ Fig. 5)")
    parser.add_argument("--no-qualitative", dest="qualitative_off", action="store_true")
    parser.add_argument("--transport-cost", dest="transport_cost", action="store_true",
                        help="Prop. 3.1 coupled vs independent cost comparison")
    parser.add_argument("--structural", action="store_true", help="Sec. 4.1 invariant checks")
    parser.add_argument("--save-samples", dest="save_samples", action="store_true")
    parser.add_argument("--trajectory", action="store_true", help="save ODE trajectory (Fig. 5)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--self-test", dest="self_test", action="store_true",
                        help="run a dependency-light sanity test and exit")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Turn parsed CLI arguments into a config override dict."""
    overrides: Dict[str, Any] = {}
    for arg_name, cfg_key in _CLI_TO_CONFIG.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[cfg_key] = value
    if getattr(args, "steps_alias", None) is not None:
        overrides["sampling_steps"] = args.steps_alias
    if getattr(args, "fid_batch_size", None) is not None:
        overrides["fid_batch_size"] = args.fid_batch_size
    return overrides


def _setup_fabric(args: argparse.Namespace, cfg: Dict[str, Any]):
    """Create a Lightning Fabric (or single-process fallback)."""
    try:
        from si.utils.distributed import setup as fabric_setup  # type: ignore

        accelerator = "gpu" if args.device and args.device.startswith("cuda") else "auto"
        return fabric_setup(accelerator=accelerator, devices="auto")
    except Exception:  # pragma: no cover - fallback
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    if args.self_test:
        return _self_test()

    overrides = config_from_args(args)
    cfg = build_config(args.config, overrides)
    if args.device:
        cfg["device"] = args.device

    outdir = args.output_dir or os.path.join(
        "eval_outputs",
        f"{cfg['task']}_{cfg.get('resolution', 256)}",
    )
    os.makedirs(outdir, exist_ok=True)

    fid_info: Optional[Dict[str, Any]] = None
    structural: Optional[Dict[str, Any]] = None
    transport: Optional[Dict[str, Any]] = None
    figures: List[str] = []
    payload: Dict[str, Any] = {}

    # ---------------------------------------------------------------- existing payload
    if args.samples:
        try:
            import torch

            payload = torch.load(args.samples, map_location="cpu")
            logger.info("loaded payload from %s", args.samples)
        except Exception as exc:
            logger.error("could not load payload '%s': %s", args.samples, exc)
            return 2

    # ---------------------------------------------------------------- generate samples
    if not payload:
        coupling = build_coupling(cfg)
        model = build_model(cfg, coupling)
        device = args.device or ("cuda" if _cuda_available() else "cpu")
        load_checkpoint(model, args.checkpoint, device=device)
        model = model.to(device)
        fabric = _setup_fabric(args, cfg)
        want_trajectory = bool(args.trajectory or cfg.get("qualitative"))
        payload = generate_samples(
            model, coupling, cfg,
            num_samples=int(cfg.get("num_samples", 50000)),
            fabric=fabric,
            synthetic=bool(args.synthetic),
            qualitative_count=int(cfg.get("qualitative_count", 8)),
            return_trajectory=want_trajectory,
        )
        payload["_model"] = model
        payload["_coupling"] = coupling

    # ---------------------------------------------------------------- FID-50k
    do_fid = bool(cfg.get("eval_fid", True)) and not args.fid_off
    if args.fid:
        do_fid = True
    if do_fid and payload.get("samples") is not None:
        fid_info = compute_fid(payload, cfg)
        if fid_info:
            logger.info("FID-50k = %s (paper: %s)", fid_info.get("fid"), fid_info.get("paper_fid"))
    elif do_fid:
        logger.warning("no samples available for FID")

    # ---------------------------------------------------------------- structural checks
    do_structural = bool(cfg.get("structural", True)) or args.structural
    if do_structural:
        structural = structural_checks(payload, cfg)
        if structural.get("available"):
            logger.info("structural check: unmasked reproduced exactly = %s",
                        structural.get("unmasked_reproduced_exactly"))

    # ---------------------------------------------------------------- Prop. 3.1
    do_transport = bool(cfg.get("transport_cost", True)) or args.transport_cost
    if do_transport:
        transport = transport_cost_comparison(cfg, synthetic=True)
        if transport.get("available"):
            logger.info("transport cost: coupled %.4f < independent %.4f (paper Prop. 3.1)",
                        transport["transport_cost_coupled"], transport["transport_cost_independent"])

    # ---------------------------------------------------------------- qualitative figures
    do_qualitative = bool(cfg.get("qualitative", True)) and not args.qualitative_off
    if args.qualitative:
        do_qualitative = True
    if do_qualitative and payload.get("qualitative"):
        figures = save_qualitative(payload, cfg, outdir)

    # ---------------------------------------------------------------- persistence + report
    if args.save_samples and payload.get("samples") is not None:
        save_samples(payload, outdir, stem=f"samples_{cfg['task']}")

    report = build_report(
        cfg, fid_info, structural, transport, figures,
        num_generated=payload.get("num_generated"),
    )
    write_report(report, outdir)
    print(format_report(report))
    return 0


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # pragma: no cover
        return False


# ======================================================================================
# Self test (dependency-light; no ImageNet / checkpoint required)
# ======================================================================================


def _self_test() -> int:
    """Smoke test the evaluation plumbing without ImageNet or a checkpoint."""
    import torch

    print("evaluate.py self-test")
    failures: List[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        status = "ok" if ok else "FAIL"
        print(f"  [{status}] {name}{(' -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    # 1) config composition + task defaults
    cfg_ip = build_config(None, {"task": "inpainting", "resolution": 256})
    check("inpainting defaults (alpha_t=t, beta_t=1-t)",
          cfg_ip["coefficients"] == "inpainting" and cfg_ip["conditioning_channels"] == 1
          and cfg_ip["mask_observed"] and cfg_ip["project_observed"])
    cfg_sr = build_config(None, {"task": "superres", "resolution": 256, "low_resolution": 64})
    check("superres defaults (gamma0, sigma>0)",
          cfg_sr["coefficients"] == "gamma0" and cfg_sr["sigma"] > 0
          and cfg_sr["conditioning_channels"] == cfg_sr["in_channels"] and not cfg_sr["mask_observed"])
    check("paper FID targets",
          PAPER_FIDS["inpainting"]["ours"] == 1.13 and PAPER_FIDS["inpainting"]["baseline"] == 1.35
          and PAPER_FIDS["superres_64_256"]["ours_train"] == 2.13
          and PAPER_FIDS["superres_64_256"]["ours_valid"] == 2.05)

    # 2) coupled base follows x0 = xi o x1 + (1-xi) o zeta and preserves observed pixels
    try:
        from si.couplings import InpaintingCoupling, SuperresCoupling

        coupling = InpaintingCoupling(sigma=1.0, num_tiles=64, missing_prob=0.3)
        x1 = torch.randn(2, 3, 32, 32)
        xi = coupling.sample_xi(x1)
        x0, xi_out = coupling.build_x0(x1, xi=xi, return_xi=True)[:2]
        obs = (xi_out > 0.5).expand_as(x1)
        check("inpainting base preserves observed pixels",
              bool(torch.allclose(x0[obs], x1[obs])) and bool(torch.all((x0[~obs] - x1[~obs]).abs() > 1e-6)))
        check("mask is binary and shared across channels",
              set(torch.unique(xi).tolist()) <= {0.0, 1.0})
        vel = torch.ones_like(x1) * 5.0
        masked = coupling.mask_velocity(vel, xi_out)
        check("velocity masked to zero on observed pixels",
              bool(masked[obs].abs().max() == 0) and bool(torch.allclose(masked[~obs], vel[~obs])))

        sr = SuperresCoupling(low_res=8, sigma=0.05)
        x1h = torch.randn(2, 3, 32, 32)
        x0s, xis = sr.build_x0(x1h, return_xi=True)[:2]
        check("superres base = U(D(x1)) + sigma zeta",
              x0s.shape == x1h.shape and xis.shape == x1h.shape and not torch.allclose(x0s, xis))
    except ImportError as exc:
        print(f"  [skip] coupling self-test (si.couplings unavailable: {exc})")

    # 3) interpolant boundary conditions support the structural claim
    try:
        from si.interpolants import Interpolant, get_coefficients

        interp = Interpolant(get_coefficients("inpainting"))
        x0 = torch.randn(4, 3, 8, 8)
        x1 = torch.randn(4, 3, 8, 8)
        t0 = torch.zeros(4)
        t1 = torch.ones(4)
        check("I_0 = x_0 and I_1 = x_1 for the inpainting preset",
              torch.allclose(interp.I_t(x0, x1, t0), x0, atol=1e-5)
              and torch.allclose(interp.I_t(x0, x1, t1), x1, atol=1e-5))
    except ImportError as exc:
        print(f"  [skip] interpolant self-test (si.interpolants unavailable: {exc})")

    # 4) report formatting picks the right table
    rep_ip = build_report(cfg_ip, {"fid": 1.20, "paper_fid": 1.13}, None, None)
    check("inpainting report builds Table 2",
          "table2" in rep_ip and "table3" not in rep_ip and len(rep_ip["table2"]["rows"]) == 2)
    rep_sr = build_report(cfg_sr, {"fid": 2.10, "paper_fid": 2.05}, None, None)
    check("superres report builds Table 3",
          "table3" in rep_sr and len(rep_sr["table3"]["rows"]) == len(REPORTED_ONLY_BASELINES) + 1)
    check("report renders without error", len(format_report(rep_sr)) > 100)

    print("-" * 60)
    if failures:
        print(f"self-test FAILED ({len(failures)}): {', '.join(failures)}")
        return 1
    print("self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
