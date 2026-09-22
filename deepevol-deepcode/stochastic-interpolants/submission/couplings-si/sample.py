"""Sampling entry point for *Stochastic Interpolants with Data-Dependent Couplings*.

This script implements **Algorithm 2** of the paper (``Sampling (via forward
Euler method)``) together with the higher-accuracy Dopri integration used for
the reported ImageNet results (Appendix B):

    Input: model ``b_hat``, corrupted sample ``m(x_1)``, ``N in N``.
    Draw noise ``zeta ~ N(0, Id)``
    Initialize ``X_0 = m(x_1) + sigma zeta``
    for n = 0, ..., N-1 do
        X_{i+1} = X_i + N^{-1} b_hat_{i/N}(X_i)
    end for
    Return: clean sample ``X_N``.

The initial condition ``X_{t=0}`` is obtained either by directly observing
``x_0 ~ rho_0(x_0)`` (e.g. a partial image) or by sampling a data point
``x_1`` and using the coupling (Eq. 18) ``x_0 = m(x_1) + sigma zeta``.

Task-specific coupling details (Sections 4.1 / 4.2):

* **In-painting** (§4.1): ``x_0 = xi o x_1 + (1 - xi) o zeta`` with a random
  64-tile mask (tile enters the mask with probability ``p = 0.3``); the model
  sees ``xi`` as appended channels of ``x``, the interpolant uses
  ``alpha_t = t``, ``beta_t = 1 - t`` and the velocity is structurally zero on
  the unmasked (observed) pixels, which we enforce by masking the network
  output; the observed pixels are projected back after every step.
* **Super-resolution** (§4.2): ``x_0 = U(D(x_1)) + sigma zeta`` with
  ``sigma > 0`` and conditioning ``xi = U(D(x_1))`` re-appended to the channel
  dimension of the velocity model input *at every integration step*.

Usage examples
--------------

    # in-painting, 256x256
    python sample.py --config configs/inpainting_256.yaml \
        --checkpoint runs/inpainting_256/model.pt --task inpainting \
        --outdir samples/inpainting_256 --num-samples 50000 --batch-size 32

    # super-resolution 64 -> 256
    python sample.py --config configs/superres_64_256.yaml \
        --checkpoint runs/superres_64_256/model.pt --task superres \
        --outdir samples/superres_64_256 --method dopri5

The script also supports computing FID-50k over the produced samples and
dumping qualitative base/model/ground-truth (Fig. 3/4/6) and probability-flow
(Fig. 5) figures.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger("si.sample")


# --------------------------------------------------------------------------- #
# Optional repo imports (kept lazy/tolerant so the module loads on its own)
# --------------------------------------------------------------------------- #
def _import_repo():
    """Import the ``si`` package pieces used at sampling time."""
    from si.couplings import InpaintingCoupling, SuperresCoupling, get_coupling
    from si.interpolants import Interpolant, get_coefficients
    from si.models import unet_from_config
    from si.samplers import (
        ODESampler,
        euler_sample,
        dopri_sample,
        probability_flow_ode,
        sample_from_coupling,
    )

    return dict(
        InpaintingCoupling=InpaintingCoupling,
        SuperresCoupling=SuperresCoupling,
        get_coupling=get_coupling,
        Interpolant=Interpolant,
        get_coefficients=get_coefficients,
        unet_from_config=unet_from_config,
        ODESampler=ODESampler,
        euler_sample=euler_sample,
        dopri_sample=dopri_sample,
        probability_flow_ode=probability_flow_ode,
        sample_from_coupling=sample_from_coupling,
    )


def _try_import(name: str):
    try:
        return __import__(name, fromlist=["*"])
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("optional import %s failed: %s", name, exc)
        return None


# --------------------------------------------------------------------------- #
# Defaults (Appendix B + addendum)
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG: Dict[str, Any] = {
    "task": "inpainting",
    "resolution": 256,
    "low_resolution": None,
    "seed": 0,
    # architecture (Appendix B)
    "model": "unet",
    "channels": 256,
    "dim_mults": (1, 1, 2, 3, 4),
    "resnet_block_groups": 8,
    "learned_sinusoidal_cond": True,
    "learned_sinusoidal_dim": 32,
    "attention_dim_head": 64,
    "attention_heads": 4,
    "random_fourier_features": False,
    "num_classes": 1000,
    "in_channels": 3,
    # coupling
    "coefficients": "inpainting",
    "sigma": 1.0,
    "num_tiles": 64,
    "missing_prob": 0.3,
    "down_mode": "area",
    "up_mode": "bilinear",
    # sampling
    "method": "dopri5",
    "steps": 50,
    "atol": 1e-5,
    "rtol": 1e-5,
    "batch_size": 32,
    "num_samples": 50_000,
    "project_observed": True,
    "return_trajectory": False,
    "num_workers": 4,
    "split": "validation",
}

# CLI argument -> config key mapping
_CLI_TO_CONFIG: Dict[str, str] = {
    "task": "task",
    "resolution": "resolution",
    "low_resolution": "low_resolution",
    "sigma": "sigma",
    "coefficients": "coefficients",
    "batch_size": "batch_size",
    "num_samples": "num_samples",
    "steps": "steps",
    "method": "method",
    "num_tiles": "num_tiles",
    "missing_prob": "missing_prob",
    "channels": "channels",
    "split": "split",
    "num_classes": "num_classes",
    "seed": "seed",
    "num_workers": "num_workers",
    "atol": "atol",
    "rtol": "rtol",
    "project_observed": "project_observed",
}


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def deep_update(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns ``base``)."""
    if not override:
        return base
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _read_yaml(path: str) -> Dict[str, Any]:
    try:
        from si.utils.config import load_config  # type: ignore

        cfg = load_config(path)
        if isinstance(cfg, dict):
            return cfg
    except Exception:  # pragma: no cover - optional dependency
        pass
    try:
        import yaml  # type: ignore

        with open(path, "r") as fh:
            loaded = yaml.safe_load(fh) or {}
        if isinstance(loaded, dict):
            return loaded
    except Exception:  # pragma: no cover - optional dependency
        pass
    with open(path, "r") as fh:
        return json.load(fh)


def _apply_task_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fill task-specific defaults for in-painting vs. super-resolution."""
    task = str(cfg.get("task", "inpainting")).lower()
    if task in ("sr", "superresolution", "super_resolution"):
        task = "superres"
    cfg["task"] = task

    if task == "inpainting":
        cfg.setdefault("coefficients", "inpainting")
        cfg.setdefault("sigma", 1.0)
        cfg.setdefault("num_tiles", 64)
        cfg.setdefault("missing_prob", 0.3)
        cfg.setdefault("conditioning_channels", cfg.get("in_channels", 3))
        cfg.setdefault("mask_observed", True)
    else:
        cfg.setdefault("coefficients", "gamma0")
        cfg.setdefault("sigma", 0.05)
        cfg.setdefault("down_mode", "area")
        cfg.setdefault("up_mode", "bilinear")
        cfg.setdefault("conditioning_channels", cfg.get("in_channels", 3))
        cfg.setdefault("mask_observed", False)
        if cfg.get("low_resolution") is None:
            res = int(cfg.get("resolution", 256))
            cfg["low_resolution"] = 64 if res <= 256 else 256
    return cfg


def build_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compose ``DEFAULT_CONFIG`` <- YAML <- explicit ``overrides``."""
    cfg: Dict[str, Any] = dict(DEFAULT_CONFIG)
    if path:
        try:
            from train import build_config as _train_build_config  # type: ignore

            cfg = _train_build_config(path, overrides or {})
            return _apply_task_defaults(cfg)
        except Exception:
            pass
        deep_update(cfg, _read_yaml(path))
    deep_update(cfg, overrides)
    return _apply_task_defaults(cfg)


# --------------------------------------------------------------------------- #
# Model / coupling construction
# --------------------------------------------------------------------------- #
def _model_kwargs(cfg: Dict[str, Any], conditioning_channels: int) -> Dict[str, Any]:
    """Assemble the U-Net (Appendix B) keyword arguments from a config."""
    res = int(cfg.get("resolution", 256))
    return dict(
        in_channels=int(cfg.get("in_channels", 3)),
        channels=int(cfg.get("channels", 256)),
        dim_mults=tuple(cfg.get("dim_mults", (1, 1, 2, 3, 4))),
        resnet_block_groups=int(cfg.get("resnet_block_groups", 8)),
        num_classes=int(cfg.get("num_classes", 1000)),
        conditioning_channels=int(conditioning_channels),
        learned_sinusoidal_cond=bool(cfg.get("learned_sinusoidal_cond", True)),
        learned_sinusoidal_dim=int(cfg.get("learned_sinusoidal_dim", 32)),
        attention_dim_head=int(cfg.get("attention_dim_head", 64)),
        attention_heads=int(cfg.get("attention_heads", 4)),
        random_fourier_features=bool(cfg.get("random_fourier_features", False)),
        mask_observed=bool(cfg.get("mask_observed", False)),
        image_size=res,
    )


def build_model(cfg: Dict[str, Any], coupling: Any = None) -> torch.nn.Module:
    """Instantiate the velocity U-Net ``b_hat_t(x, xi)``."""
    repo = _import_repo()
    in_channels = int(cfg.get("in_channels", 3))
    conditioning_channels = int(cfg.get("conditioning_channels", in_channels))
    kwargs = _model_kwargs(cfg, conditioning_channels)
    model = repo["unet_from_config"](kwargs)
    logger.info(
        "built velocity U-Net: %d input channels (x_t %d + xi %d), %s params",
        in_channels + conditioning_channels,
        in_channels,
        conditioning_channels,
        f"{sum(p.numel() for p in model.parameters()):,}",
    )
    return model


def build_coupling(cfg: Dict[str, Any], repo: Optional[Dict[str, Any]] = None):
    """Construct the task coupling from the config."""
    repo = repo or _import_repo()
    task = cfg.get("task", "inpainting")
    if task == "inpainting":
        return repo["InpaintingCoupling"](
            sigma=float(cfg.get("sigma", 1.0)),
            num_tiles=int(cfg.get("num_tiles", 64)),
            missing_prob=float(cfg.get("missing_prob", 0.3)),
            coefficients=str(cfg.get("coefficients", "inpainting")),
        )
    return repo["SuperresCoupling"](
        low_res=int(cfg.get("low_resolution", 64)),
        sigma=float(cfg.get("sigma", 0.05)),
        down_mode=str(cfg.get("down_mode", "area")),
        up_mode=str(cfg.get("up_mode", "bilinear")),
        coefficients=str(cfg.get("coefficients", "gamma0")),
    )


def build_interpolant(cfg: Dict[str, Any], repo: Optional[Dict[str, Any]] = None):
    repo = repo or _import_repo()
    return repo["Interpolant"](repo["get_coefficients"](str(cfg.get("coefficients", "gamma0"))))


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #
def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    map_location: Any = None,
    use_ema: bool = True,
    strict: bool = False,
) -> Dict[str, Any]:
    """Load weights (preferring EMA when present) and return the checkpoint dict."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        from train import Trainer  # type: ignore

        return Trainer.load_into(model, path, map_location=map_location, use_ema=use_ema)
    except Exception:
        pass

    ckpt = torch.load(path, map_location=map_location or "cpu")
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("ema_model", "model_ema", "ema", "model", "state_dict", "module"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                if key in ("model", "state_dict", "module", "ema_model", "model_ema", "ema"):
                    break
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        logger.warning("checkpoint load: %d missing / %d unexpected keys", len(missing), len(unexpected))
    logger.info("loaded checkpoint %s", path)
    return ckpt if isinstance(ckpt, dict) else {"state_dict": state}


# --------------------------------------------------------------------------- #
# Task-specific draw of x_0 and conditioning xi
# --------------------------------------------------------------------------- #
def draw_task_inputs(
    coupling: Any,
    cfg: Dict[str, Any],
    x1: torch.Tensor,
    low: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``(x0, xi, zeta)`` for the current task.

    * In-painting (§4.1): ``x0 = xi o x1 + (1 - xi) o zeta``, ``xi`` the mask.
    * Super-resolution (§4.2): ``x0 = U(D(x1)) + sigma zeta``, ``xi = U(D(x1))``.
    """
    if isinstance(coupling, type(coupling)) and getattr(coupling, "name", "") == "inpainting":
        pass
    zeta = None
    x0, xi = coupling.build_x0(x1, zeta=zeta, generator=generator, return_xi=True)
    return x0, xi, zeta


def _mask_fn_for(coupling: Any, cfg: Dict[str, Any]):
    """Structural velocity masking hook for in-painting (§4.1)."""
    if cfg.get("task") == "inpainting" and hasattr(coupling, "mask_velocity"):
        return coupling.mask_velocity
    return None


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_batch(
    model: torch.nn.Module,
    coupling: Any,
    cfg: Dict[str, Any],
    x1: torch.Tensor,
    low: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    return_base: bool = False,
    return_trajectory: Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    """Integrate the probability-flow ODE for one batch (Algorithm 2 / Dopri)."""
    repo = _import_repo()
    x1 = x1.to(dtype=torch.float32)

    # --- x_0 and conditioning -------------------------------------------------
    if cfg.get("task") == "superres":
        low = coupling.D(x1) if low is None else low
        xi = coupling.U(low, high_res=x1.shape[-2:])
        x0 = coupling.add_noise(xi, float(cfg.get("sigma", 0.05)), generator=generator)
    else:
        x0, xi = coupling.build_x0(x1, generator=generator, return_xi=True)

    mask_fn = _mask_fn_for(coupling, cfg)
    project = cfg.get("project_observed", True) if cfg.get("task") == "inpainting" else False

    method = str(cfg.get("method", "dopri5"))
    steps = int(cfg.get("steps", 50))
    traj = cfg.get("return_trajectory", False) if return_trajectory is None else return_trajectory

    kwargs = dict(
        xi=xi,
        y=labels,
        mask_fn=mask_fn,
        return_trajectory=traj,
    )
    if project and cfg.get("task") == "inpainting":
        kwargs.update(x1=x1, observed_value=1.0)

    if method in ("dopri5", "dopri8", "bosh3", "fehlberg2", "adaptive_heun"):
        kwargs.update(atol=float(cfg.get("atol", 1e-5)), rtol=float(cfg.get("rtol", 1e-5)))
        kwargs["trajectory_steps"] = steps
    else:
        kwargs["steps"] = steps

    out = repo["probability_flow_ode"](model, x0, method=method, **kwargs)
    if isinstance(out, tuple):
        sample, trajectory = out
    else:
        sample, trajectory = out, None

    result: Dict[str, torch.Tensor] = {
        "sample": sample,
        "x0": x0,
        "x1": x1,
        "xi": xi,
    }
    if labels is not None:
        result["label"] = labels
    if trajectory is not None:
        result["trajectory"] = trajectory
    if return_base:
        result["base"] = x0
    return result


def _infinite_batches(loader):
    while True:
        for batch in loader:
            yield batch


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract ``(x1, labels, low_res)`` from a dataset batch."""
    labels = low = None
    if isinstance(batch, dict):
        x1 = batch.get("x1", batch.get("image", batch.get("images")))
        labels = batch.get("label", batch.get("y"))
        low = batch.get("low", batch.get("low_res"))
    elif isinstance(batch, (tuple, list)):
        x1 = batch[0]
        if len(batch) == 2:
            labels = batch[1] if torch.is_tensor(batch[1]) or batch[1] is not None else None
            if isinstance(batch[1], (list, tuple)):
                labels, low = batch[1][0], batch[1][1] if len(batch[1]) > 1 else None
        elif len(batch) >= 3:
            low, labels = batch[1], batch[2]
    else:
        x1 = batch
    if x1 is None:
        raise ValueError("could not find images in batch")
    return x1, labels, low


def build_loader(
    cfg: Dict[str, Any],
    num_samples: int,
    batch_size: int,
    fabric: Any = None,
    split: Optional[str] = None,
    synthetic: bool = False,
):
    """ImageNet loader sized for ``num_samples`` generations."""
    from si.data import build_dataloader as _build_dataloader

    steps = max(1, math.ceil(num_samples / batch_size))
    return _build_dataloader(
        split=split or cfg.get("split", "validation"),
        batch_size=batch_size,
        resolution=int(cfg.get("resolution", 256)),
        low_resolution=cfg.get("low_resolution"),
        num_workers=int(cfg.get("num_workers", 4)),
        shuffle=True,
        drop_last=False,
        synthetic=synthetic,
        max_samples=max(num_samples, steps * batch_size),
        return_dict=True,
    )


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    coupling: Any,
    cfg: Dict[str, Any],
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    loader: Any = None,
    fabric: Any = None,
    synthetic: bool = False,
    progress: bool = True,
    qualitative_count: int = 4,
) -> Dict[str, Any]:
    """Generate ``num_samples`` samples, optionally 50k for FID (§4.1, §4.2)."""
    num_samples = int(num_samples or cfg.get("num_samples", 50_000))
    batch_size = int(batch_size or cfg.get("batch_size", 32))
    if loader is None:
        loader = build_loader(cfg, num_samples, batch_size, fabric=fabric, synthetic=synthetic)

    device = _device_of(model)
    model.eval()

    samples: List[torch.Tensor] = []
    bases: List[torch.Tensor] = []
    ground_truths: List[torch.Tensor] = []
    conds: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []
    trajectories: Optional[torch.Tensor] = None

    made = 0
    batches = _infinite_batches(loader)
    t_start = time.time()
    while made < num_samples:
        batch = next(batches)
        x1, labels, low = unpack_batch(batch)
        take = min(batch_size, num_samples - made)
        x1 = x1[:take]
        labels = labels[:take] if labels is not None else None
        low = low[:take] if low is not None else None
        x1 = x1.to(device)

        want_traj = bool(cfg.get("save_trajectory", False)) and trajectories is None
        out = sample_batch(
            model,
            coupling,
            cfg,
            x1,
            low=low,
            labels=labels,
            return_trajectory=want_traj,
        )

        samples.append(out["sample"].detach().cpu())
        bases.append(out["x0"].detach().cpu())
        ground_truths.append(out["x1"].detach().cpu())
        conds.append(out["xi"].detach().cpu())
        if "label" in out:
            labels_all.append(out["label"].detach().cpu())
        if "trajectory" in out:
            trajectories = out["trajectory"].detach().cpu()

        made += out["sample"].shape[0]
        if progress:
            elapsed = time.time() - t_start
            rate = made / max(elapsed, 1e-6)
            logger.info(
                "sampled %d/%d (%.1f it/s, %.1f s elapsed)",
                made,
                num_samples,
                rate,
                elapsed,
            )

    payload: Dict[str, Any] = {
        "samples": torch.cat(samples, dim=0)[:num_samples],
        "base": torch.cat(bases, dim=0)[:num_samples],
        "ground_truth": torch.cat(ground_truths, dim=0)[:num_samples],
        "xi": torch.cat(conds, dim=0)[:num_samples],
    }
    if labels_all:
        payload["labels"] = torch.cat(labels_all, dim=0)[:num_samples]
    if trajectories is not None:
        payload["trajectory"] = trajectories

    # keep a few triples for qualitative figures (Figs. 3/4/6)
    n = min(int(qualitative_count), payload["samples"].shape[0])
    payload["qualitative"] = {
        "base": payload["base"][:n],
        "model": payload["samples"][:n],
        "ground_truth": payload["ground_truth"][:n],
        "xi": payload["xi"][:n],
    }
    return payload


def _device_of(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - parameterless model
        return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Evaluation / saving
# --------------------------------------------------------------------------- #
def compute_fid(payload: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[Any]:
    """Compute FID-50k over generated samples (Table 2 / Table 3)."""
    try:
        from eval.fid import evaluate_fid_50k
    except Exception as exc:  # pragma: no cover - optional evaluation deps
        logger.warning("FID evaluation unavailable: %s", exc)
        return None
    task = cfg.get("task", "inpainting")
    if task == "superres":
        task_key = f"superres_{cfg.get('low_resolution', 64)}_{cfg.get('resolution', 256)}"
    else:
        task_key = "inpainting"
    try:
        result = evaluate_fid_50k(
            payload["samples"],
            task=task_key,
            batch_size=int(cfg.get("batch_size", 32)),
            num_samples=int(payload["samples"].shape[0]),
            resolution=int(cfg.get("resolution", 256)),
        )
        logger.info("FID-50k: %s", result.as_dict() if hasattr(result, "as_dict") else result)
        return result
    except Exception as exc:  # pragma: no cover
        logger.warning("FID computation failed: %s", exc)
        return None


def save_qualitative(payload: Dict[str, Any], cfg: Dict[str, Any], outdir: str) -> List[str]:
    """Write Fig. 3/4/6-style base/model/GT triples."""
    written: List[str] = []
    try:
        from eval.qualitative import make_triples, save_triples_grid
    except Exception as exc:  # pragma: no cover - optional matplotlib deps
        logger.warning("qualitative figures unavailable: %s", exc)
        return written

    qual = payload.get("qualitative")
    if not qual:
        return written
    labels = payload.get("labels")
    try:
        triples = make_triples(
            qual["base"],
            qual["model"],
            qual["ground_truth"],
            xi=qual.get("xi"),
            class_labels=labels[: qual["model"].shape[0]] if labels is not None else None,
        )
        task = cfg.get("task", "inpainting")
        path = os.path.join(
            outdir,
            "triples_inpainting.png" if task == "inpainting" else "triples_superres.png",
        )
        save_triples_grid(
            triples,
            path,
            title=("In-painting (ImageNet)" if task == "inpainting" else "Super-resolution (ImageNet)"),
        )
        written.append(path)
    except Exception as exc:  # pragma: no cover
        logger.warning("saving triples failed: %s", exc)

    if "trajectory" in payload and payload["trajectory"] is not None:
        try:
            from eval.qualitative import make_probability_flow_figure

            traj = payload["trajectory"]
            first = traj[:, 0] if traj.dim() == 5 else traj
            path = os.path.join(outdir, "probability_flow.png")
            make_probability_flow_figure(
                first,
                ground_truth=payload["ground_truth"][: first.shape[0]],
                mask=payload["xi"][: first.shape[0]],
                path=path,
                title="Probability-flow ODE (Fig. 5)",
            )
            written.append(path)
        except Exception as exc:  # pragma: no cover
            logger.warning("probability-flow figure failed: %s", exc)
    return written


def save_samples(payload: Dict[str, Any], outdir: str, stem: str = "samples") -> str:
    """Persist generated tensors as a ``.pt`` bundle."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{stem}.pt")
    torch.save({k: v for k, v in payload.items() if k != "qualitative"}, path)
    logger.info("wrote %s (%d samples)", path, payload["samples"].shape[0])
    return path


def save_samples_png(payload: Dict[str, Any], outdir: str, max_images: int = 64, nrow: int = 8) -> Optional[str]:
    """Optionally dump generated samples as a PNG grid (qualitative inspection)."""
    try:
        from eval.qualitative import save_image_grid
    except Exception:  # pragma: no cover
        return None
    try:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "samples_grid.png")
        save_image_grid(payload["samples"][:max_images], path, nrow=nrow, title="Model samples")
        return path
    except Exception as exc:  # pragma: no cover
        logger.warning("saving sample grid failed: %s", exc)
        return None


def validate_inpainting(payload: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, float]:
    """Check the §4.1 invariant ``xi o X_{t=1} = xi o x_1`` (observed pixels fixed)."""
    if cfg.get("task") != "inpainting" or "xi" not in payload:
        return {}
    try:
        from eval.qualitative import verify_inpainting_consistency

        stats = verify_inpainting_consistency(
            payload["base"], payload["samples"], payload["ground_truth"], payload["xi"]
        )
        logger.info("in-painting consistency: %s", stats)
        return stats
    except Exception as exc:  # pragma: no cover
        logger.debug("consistency check failed: %s", exc)
        return {}
    return {}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample from a trained stochastic-interpolant model (Algorithm 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, required=False, default=None, help="trained model checkpoint")
    parser.add_argument("--task", type=str, default=None, choices=["inpainting", "superres"])
    parser.add_argument("--outdir", type=str, default="samples", help="output directory")
    parser.add_argument("--num-samples", type=int, default=None, help="number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--method", type=str, default=None, help="dopri5 | euler | rk4 | ...")
    parser.add_argument("--steps", type=int, default=None, help="Euler steps N (Algorithm 2)")
    parser.add_argument("--resolution", type=int, default=None, choices=[256, 512])
    parser.add_argument("--low-resolution", type=int, default=None, choices=[64, 256])
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--coefficients", type=str, default=None)
    parser.add_argument("--num-tiles", type=int, default=None)
    parser.add_argument("--missing-prob", type=float, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--no-project-observed", action="store_true", help="disable fixed-pixel projection")
    parser.add_argument("--fid", action="store_true", help="compute FID-50k on the generated samples")
    parser.add_argument("--qualitative", action="store_true", help="save base/model/GT triples")
    parser.add_argument("--save-trajectory", action="store_true", help="store ODE trajectory (Fig. 5)")
    parser.add_argument("--save-png", action="store_true", help="dump a PNG grid of the samples")
    parser.add_argument("--synthetic", action="store_true", help="use the synthetic dataset (smoke test)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--logger", type=str, default="info")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for arg, key in _CLI_TO_CONFIG.items():
        value = getattr(args, arg.replace("-", "_"), None)
        if value is not None:
            overrides[key] = value
    if args.no_project_observed:
        overrides["project_observed"] = False
    if args.save_trajectory:
        overrides["save_trajectory"] = True
    return overrides


def _setup_fabric(args: argparse.Namespace, cfg: Dict[str, Any]):
    try:
        from si.utils import distributed as dist_utils  # type: ignore

        kwargs: Dict[str, Any] = {}
        if args.device:
            kwargs["accelerator"] = args.device
        fabric = dist_utils.setup(**kwargs)
        return fabric
    except Exception as exc:  # pragma: no cover - fallback path
        logger.debug("fabric setup failed (%s); running single process", exc)
        return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.logger).upper(), logging.INFO),
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    )

    cfg = build_config(args.config, config_from_args(args))
    if not args.checkpoint:
        logger.warning("no --checkpoint given: sampling from an untrained model")
    os.makedirs(args.outdir, exist_ok=True)

    if cfg.get("seed") is not None:
        torch.manual_seed(int(cfg["seed"]))

    repo = _import_repo()
    coupling = build_coupling(cfg, repo)
    model = build_model(cfg, coupling)

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, use_ema=True)

    fabric = _setup_fabric(args, cfg)
    if fabric is not None:
        try:
            model = fabric.setup_module(model)
        except Exception:  # pragma: no cover
            pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fabric is None:
        model = model.to(device)

    payload = generate_samples(
        model,
        coupling,
        cfg,
        num_samples=cfg.get("num_samples"),
        batch_size=cfg.get("batch_size"),
        fabric=fabric,
        synthetic=args.synthetic,
    )

    save_samples(payload, args.outdir, stem=f"seeded_{cfg.get('seed', 0)}")
    if args.save_png:
        save_samples_png(payload, args.outdir)

    summary: Dict[str, Any] = {
        "task": cfg.get("task"),
        "resolution": cfg.get("resolution"),
        "low_resolution": cfg.get("low_resolution"),
        "method": cfg.get("method"),
        "steps": cfg.get("steps"),
        "num_samples": int(payload["samples"].shape[0]),
        "sigma": cfg.get("sigma"),
        "coefficients": cfg.get("coefficients"),
    }

    consistency = validate_inpainting(payload, cfg)
    if consistency:
        summary["inpainting_consistency"] = consistency

    if args.qualitative:
        written = save_qualitative(payload, cfg, args.outdir)
        summary["figures"] = written

    if args.fid:
        result = compute_fid(payload, cfg)
        if result is not None:
            summary["fid"] = result.as_dict() if hasattr(result, "as_dict") else str(result)

    with open(os.path.join(args.outdir, "sample_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("done: %s", json.dumps(summary, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
