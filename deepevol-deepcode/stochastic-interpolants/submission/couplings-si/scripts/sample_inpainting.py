#!/usr/bin/env python
"""In-painting sampling entry point for *Stochastic Interpolants with Data-Dependent Couplings*.

This script implements sampling for the ImageNet in-painting experiment of Section 4.1 using
Algorithm 2 (probability-flow ODE / forward Euler, also Dopri):

* the coupling sets ``x_0 = xi o x_1 + (1 - xi) o zeta`` with ``xi in {0,1}^{C x W x H}`` the
  missingness mask (1 = observed/kept from ``x_1``, 0 = missing/replaced by ``zeta ~ N(0, Id)``),
  the same mask value for all channels at a given spatial location (Section 4.1);
* the interpolant uses ``alpha_t = t``, ``beta_t = 1 - t`` (so ``I_0 = x_0`` corrupted and
  ``I_1 = x_1`` clean, both in the repository convention ``t = 0 -> t = 1``);
* the model is the Appendix-B U-Net, conditioned on the mask ``xi`` appended as extra input
  channels and on the ImageNet class label;
* because ``xi o I_t = xi o x_1`` for every ``t``, the velocity field vanishes on observed pixels:
  we both mask the network output (``mask_fn``) and optionally project the observed pixels back
  after every integration step (``project``).

Reported reference numbers (Table 2): FID-50k = 1.13 for the dependent coupling (ours) versus
1.35 for the uncoupled interpolant baseline (independent Gaussian base).

Examples
--------
In-painting at 256x256 with Dopri (paper settings), 50k samples for FID-50k::

    python scripts/sample_inpainting.py --config configs/inpainting_256.yaml \\
        --checkpoint runs/inpainting256/last.pt --resolution 256 \\
        --num-samples 50000 --method dopri5 --fid --qualitative --outdir samples/inpainting256

In-painting at 512x512 with the 64-tile random mask (p = 0.3) and forward Euler, N = 250::

    python scripts/sample_inpainting.py --config configs/inpainting_512.yaml \\
        --checkpoint runs/inpainting512/last.pt --resolution 512 \\
        --num-tiles 64 --missing-prob 0.3 --method euler --steps 250 --qualitative

A fixed (pre-specified) mask, as in the paper's "given a pre-specified mask" setting::

    python scripts/sample_inpainting.py --checkpoint ckpt.pt --mask-file mask.npy --qualitative

Smoke test without ImageNet (synthetic tensors, tiny U-Net)::

    python scripts/sample_inpainting.py --self-test
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Make the repository root importable when this file is executed as a script.
# --------------------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = logging.getLogger("sample_inpainting")

# --------------------------------------------------------------------------------------
# Defaults (Section 4.1 + Appendix B + addendum)
# --------------------------------------------------------------------------------------
DEFAULT_INPAINTING_CONFIG: Dict[str, Any] = {
    "task": "inpainting",
    "resolution": 256,
    "channels": 3,
    "num_classes": 1000,
    # Section 4.1: alpha_t = t, beta_t = 1 - t (so I_0 = x_0 corrupted, I_1 = x_1 clean).
    "coefficients": "inpainting",
    "sigma": 1.0,
    # Section 4.1: image tiled into 64 tiles, each tile enters the mask with probability p = 0.3.
    "num_tiles": 64,
    "missing_prob": 0.3,
    "expand_channels": False,
    # Structural velocity masking + per-step projection onto the observed pixels.
    "mask_observed": True,
    "project_observed": True,
    "conditioning_channels": 1,
    "base_mode": "coupled",
    # Sampling (Algorithm 2 / Appendix B).
    "method": "dopri5",
    "steps": 50,
    "atol": 1e-5,
    "rtol": 1e-5,
    "batch_size": 32,
    "num_samples": 50000,
    "num_workers": 4,
    "split": "validation",
    "synthetic": False,
    "seed": 0,
    "qualitative_count": 8,
    "model": {
        "channels": 256,
        "dim_mults": (1, 1, 2, 3, 4),
        "resnet_block_groups": 8,
        "learned_sinusoidal_cond": True,
        "learned_sinusoidal_dim": 32,
        "random_fourier_features": False,
        "attention_dim_head": 64,
        "attention_heads": 4,
        "num_classes": 1000,
        "class_dropout_prob": 0.1,
    },
}

# Table 2 reference numbers.
PAPER_FID_50K = 1.13
BASELINE_FID_50K = 1.35

ADAPTIVE_METHODS = ("dopri5", "dopri8", "bosh3", "fehlberg2", "adaptive_heun")

__all__ = [
    "DEFAULT_INPAINTING_CONFIG",
    "PAPER_FID_50K",
    "BASELINE_FID_50K",
    "build_config",
    "build_coupling",
    "build_interpolant",
    "build_model",
    "load_checkpoint",
    "make_mask",
    "load_mask_file",
    "sample_batch_with_mask",
    "generate_samples",
    "check_inpainting_consistency",
    "save_qualitative",
    "save_samples",
    "compute_fid",
    "parse_args",
    "config_from_args",
    "main",
]


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------
def deep_update(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        elif value is not None:
            out[key] = value
    return out


def _require(module_name: str, attr: Optional[str] = None) -> Any:
    """Import ``module_name`` lazily, raising a readable error on failure."""
    import importlib

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - import error path
        raise ImportError(f"could not import '{module_name}': {exc}") from exc
    if attr is None:
        return module
    if not hasattr(module, attr):
        raise ImportError(f"'{module_name}' has no attribute '{attr}'")
    return getattr(module, attr)


def make_generator(seed: Optional[int], device: Any = None):
    """Create a ``torch.Generator`` on ``device`` with the given ``seed`` (or ``None``)."""
    import torch

    generator = torch.Generator(device=device) if device is not None else torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))
    return generator


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
def build_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compose in-painting defaults <- YAML file <- explicit overrides."""
    cfg: Dict[str, Any] = dict(DEFAULT_INPAINTING_CONFIG)
    file_cfg: Dict[str, Any] = {}
    if path:
        file_cfg = _read_yaml(path)
    cfg = deep_update(cfg, file_cfg)
    cfg = deep_update(cfg, overrides)
    return resolve_task_settings(cfg)


def _read_yaml(path: str) -> Dict[str, Any]:
    """Read a YAML config, falling back to the repo loader and then JSON."""
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return dict(payload or {})
    except Exception:  # pragma: no cover - fallback path
        try:
            loader = _require("si.utils.config", "load_config")
            return dict(loader(path) or {})
        except Exception:
            with open(path, "r", encoding="utf-8") as handle:
                return dict(json.load(handle) or {})


def resolve_task_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Force the in-painting task conventions from Section 4.1 onto ``cfg``."""
    cfg = dict(cfg)
    cfg["task"] = "inpainting"
    model_cfg = dict(cfg.get("model") or {})
    base_mode = str(cfg.get("base_mode", "coupled")).lower()
    independent = base_mode in ("independent", "uncoupled", "gaussian")

    if independent:
        # Baseline of Table 2: rho_0 is a Gaussian with independent coupling to rho_1, i.e. the
        # base sample carries no information about x_1 at all.
        cfg.setdefault("coefficients", "gamma0")
        cfg["mask_observed"] = False
        cfg["project_observed"] = False
    else:
        # Section 4.1: alpha_t = t, beta_t = 1 - t, gamma_t = 0.
        cfg["coefficients"] = cfg.get("coefficients") or "inpainting"
        if cfg["coefficients"] in ("gamma0", "superres"):
            cfg["coefficients"] = "inpainting"

    model_cfg["in_channels"] = int(cfg.get("channels", 3))
    model_cfg["out_channels"] = int(cfg.get("channels", 3))
    model_cfg["image_size"] = int(cfg.get("resolution", 256))
    model_cfg["num_classes"] = int(cfg.get("num_classes", model_cfg.get("num_classes", 1000)))
    # The mask xi is appended to the channel dimension of x_t (Section 4.1).
    model_cfg["conditioning_channels"] = int(cfg.get("conditioning_channels", 1))
    model_cfg["mask_observed"] = bool(cfg.get("mask_observed", True))
    if isinstance(model_cfg.get("dim_mults"), tuple):
        model_cfg["dim_mults"] = list(model_cfg["dim_mults"])
    cfg["model"] = model_cfg
    return cfg


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------
def build_coupling(cfg: Dict[str, Any], device: Any = None):
    """Build the in-painting coupling (``x_0 = xi o x_1 + (1 - xi) o zeta``)."""
    try:
        coupling_cls = _require("si.couplings", "InpaintingCoupling")
    except ImportError:  # pragma: no cover
        get_coupling = _require("si.couplings", "get_coupling")
        return get_coupling(
            "inpainting",
            sigma=float(cfg.get("sigma", 1.0)),
            num_tiles=int(cfg.get("num_tiles", 64)),
            missing_prob=float(cfg.get("missing_prob", 0.3)),
            expand_channels=bool(cfg.get("expand_channels", False)),
        )
    coupling = coupling_cls(
        sigma=float(cfg.get("sigma", 1.0)),
        num_tiles=int(cfg.get("num_tiles", 64)),
        missing_prob=float(cfg.get("missing_prob", 0.3)),
        expand_channels=bool(cfg.get("expand_channels", False)),
        coefficients=str(cfg.get("coefficients", "inpainting")),
    )
    if device is not None and hasattr(coupling, "to"):
        try:
            coupling = coupling.to(device)
        except Exception:  # pragma: no cover - coupling need not be a module
            pass
    return coupling


def build_interpolant(cfg: Dict[str, Any]):
    """Build the interpolant for ``cfg['coefficients']``."""
    Interpolant = _require("si.interpolants", "Interpolant")
    get_coefficients = _require("si.interpolants", "get_coefficients")
    return Interpolant(get_coefficients(str(cfg.get("coefficients", "inpainting"))))


def build_model(cfg: Dict[str, Any], coupling: Any = None):
    """Instantiate the Appendix-B velocity U-Net from ``cfg['model']``."""
    model_cfg = dict(cfg.get("model") or {})
    if coupling is not None and hasattr(coupling, "in_channels"):
        try:
            model_cfg["in_channels"] = int(coupling.in_channels(int(cfg.get("channels", 3))))
            model_cfg.pop("conditioning_channels", None)
        except Exception:  # pragma: no cover
            pass
    try:
        unet_from_config = _require("si.models", "unet_from_config")
        return unet_from_config(model_cfg)
    except ImportError:  # pragma: no cover
        VelocityUNet = _require("si.models", "VelocityUNet")
        return VelocityUNet(**model_cfg)


def load_checkpoint(
    model: Any,
    path: Optional[str],
    device: Any = None,
    use_ema: bool = True,
) -> Dict[str, Any]:
    """Load a training checkpoint into ``model`` (prefers EMA weights when present)."""
    import torch

    if not path:
        logger.warning("no --checkpoint given: sampling from randomly initialised weights")
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device or "cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    ema_state = None
    if isinstance(ckpt, dict) and use_ema:
        for key in ("ema", "ema_state_dict", "model_ema"):
            if isinstance(ckpt.get(key), dict):
                ema_state = ckpt[key]
                break
    if ema_state is not None:
        inner = ema_state.get("shadow") if "shadow" in ema_state else ema_state
        if isinstance(inner, dict) and inner:
            state = inner
            logger.info("using EMA weights from %s", path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("%d missing keys when loading checkpoint", len(missing))
    if unexpected:
        logger.warning("%d unexpected keys when loading checkpoint", len(unexpected))
    return ckpt if isinstance(ckpt, dict) else {}


# --------------------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------------------
def make_mask(
    shape: Any,
    num_tiles: int = 64,
    missing_prob: float = 0.3,
    mode: str = "random",
    generator: Any = None,
    device: Any = None,
) -> Any:
    """Build a missingness mask with the repo convention (1 = observed, 0 = missing).

    ``mode`` is one of ``"random"`` (Section 4.1 training/eval masks: 64 tiles, p = 0.3),
    ``"center"`` (a pre-specified central square), ``"zeros"`` (nothing observed, pure noise
    initialisation) or ``"ones"`` (everything observed).
    """
    import torch

    random_tile_mask = _require("si.couplings", "random_tile_mask")
    if isinstance(shape, torch.Tensor):
        height, width = int(shape.shape[-2]), int(shape.shape[-1])
    elif isinstance(shape, (tuple, list, torch.Size)):
        height, width = int(shape[-2]), int(shape[-1])
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported mask shape spec: {shape!r}")

    if mode == "center":
        mask = torch.zeros(1, 1, height, width, device=device)
        h0, h1 = height // 4, 3 * height // 4
        w0, w1 = width // 4, 3 * width // 4
        mask[..., h0:h1, w0:w1] = 1.0
        return mask
    if mode == "zeros":
        return torch.zeros(1, 1, height, width, device=device)
    if mode == "ones":
        return torch.ones(1, 1, height, width, device=device)
    return random_tile_mask(
        (1, 1, height, width),
        num_tiles=int(num_tiles),
        missing_prob=float(missing_prob),
        generator=generator,
        device=device,
    )


def load_mask_file(path: str, device: Any = None) -> Any:
    """Load a pre-specified (fixed) mask from ``.npy`` / ``.pt`` / ``.pth``."""
    import torch

    if path.endswith(".npy"):
        import numpy as np

        array = np.load(path)
        mask = torch.from_numpy(array)
    else:
        mask = torch.load(path, map_location="cpu")
        if isinstance(mask, dict):
            for key in ("xi", "mask", "masks"):
                if key in mask:
                    mask = mask[key]
                    break
    mask = torch.as_tensor(mask).float()
    while mask.dim() < 4:
        mask = mask.unsqueeze(0)
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    return mask.to(device) if device is not None else mask


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------
def _as_mask_fn(coupling: Any, xi: Any) -> Optional[Callable]:
    """Adapt ``coupling.mask_velocity`` to a single-argument ``mask_fn(velocity)`` closure."""
    fn = getattr(coupling, "mask_velocity", None)
    if fn is None:
        return None
    try:
        params = [p for p in inspect.signature(fn).parameters.values()]
        required = [p for p in params if p.default is inspect.Parameter.empty]
    except (TypeError, ValueError):  # pragma: no cover - builtins
        params, required = [], []

    if len(params) <= 1:
        def mask_fn(velocity, *args, **kwargs):  # type: ignore[override]
            return fn(velocity)
    else:
        def mask_fn(velocity, *args, **kwargs):  # type: ignore[override]
            return fn(velocity, xi)
    return mask_fn


def sample_batch_with_mask(
    model: Any,
    coupling: Any,
    cfg: Dict[str, Any],
    x1: Any,
    xi: Any = None,
    labels: Any = None,
    generator: Any = None,
    return_trajectory: bool = False,
    base_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Sample one batch: build ``x_0`` (coupled or independent) then integrate the ODE.

    Implements Algorithm 2 with the initial condition ``X_0 = m(x_1) + sigma zeta`` and the
    probability-flow ODE ``X_dot = b_hat_t(X_t, xi)`` integrated from ``t = 0`` to ``t = 1``.
    """
    import torch

    probability_flow_ode = _require("si.samplers", "probability_flow_ode")
    make_project_fn = _require("si.samplers", "make_project_fn")

    base_mode = str(base_mode or cfg.get("base_mode", "coupled")).lower()
    independent = base_mode in ("independent", "uncoupled", "gaussian")

    if xi is None:
        if independent:
            xi = torch.zeros(x1.shape[0], 1, x1.shape[-2], x1.shape[-1], device=x1.device, dtype=x1.dtype)
        elif hasattr(coupling, "sample_xi"):
            xi = coupling.sample_xi(x1, generator=generator)
        else:  # pragma: no cover - coupling API is guaranteed by the repo
            xi = None

    zeta = torch.randn(x1.shape, generator=generator, device=x1.device, dtype=x1.dtype)
    if independent:
        # rho_0 = N(0, Id) with no dependence on x_1 (Table 2 baseline).
        x0 = zeta
        xi_model = xi
    else:
        x0, xi_model = coupling.build_x0(x1, xi=xi, zeta=zeta, generator=generator, return_xi=True)

    mask_fn = _as_mask_fn(coupling, xi_model) if cfg.get("mask_observed", True) else None
    project = None
    if cfg.get("project_observed", True) and xi_model is not None:
        try:
            project = make_project_fn(xi=xi_model, x1=x1, observed_value=1.0)
        except Exception:  # pragma: no cover - projection is a safety net only
            project = None

    method = str(cfg.get("method", "dopri5"))
    common: Dict[str, Any] = {
        "xi": xi_model,
        "y": labels,
        "method": method,
        "mask_fn": mask_fn,
        "project": project,
        "return_trajectory": bool(return_trajectory),
    }
    if method in ADAPTIVE_METHODS:
        common["atol"] = float(cfg.get("atol", 1e-5))
        common["rtol"] = float(cfg.get("rtol", 1e-5))
        common["trajectory_steps"] = int(cfg.get("steps", 50))
    else:
        common["steps"] = int(cfg.get("steps", 50))

    out = probability_flow_ode(model, x0, **common)
    trajectory = None
    if isinstance(out, (tuple, list)):
        sample, trajectory = out[0], out[1]
    else:
        sample = out
    result: Dict[str, Any] = {"sample": sample, "x0": x0, "x1": x1, "xi": xi_model, "zeta": zeta}
    if labels is not None:
        result["label"] = labels
    if trajectory is not None:
        result["trajectory"] = trajectory
    return result


def unpack_batch(batch: Any) -> Tuple[Any, Any, Any]:
    """Extract ``(x1, labels, low_res)`` from a dict or tuple batch."""
    if isinstance(batch, dict):
        x1 = batch.get("x1", batch.get("image", batch.get("images")))
        labels = batch.get("label", batch.get("labels", batch.get("y")))
        low = batch.get("low", batch.get("low_res"))
        return x1, labels, low
    if isinstance(batch, (tuple, list)):
        if len(batch) == 1:
            return batch[0], None, None
        if len(batch) == 2:
            return batch[0], batch[1], None
        return batch[0], batch[1], batch[2]
    return batch, None, None  # pragma: no cover - defensive


def build_loader(cfg: Dict[str, Any], num_samples: int, batch_size: int, synthetic: bool = False):
    """Build the ImageNet loader used to obtain the ``x_1`` targets (and ``xi`` context)."""
    build_dataloader = _require("si.data", "build_dataloader")
    kwargs: Dict[str, Any] = dict(
        split=str(cfg.get("split", "validation")),
        batch_size=int(batch_size),
        resolution=int(cfg.get("resolution", 256)),
        num_workers=int(cfg.get("num_workers", 4)),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        synthetic=bool(synthetic or cfg.get("synthetic", False)),
        synthetic_length=int(max(num_samples, 8)),
        max_samples=int(num_samples),
        return_dict=True,
    )
    if cfg.get("cache_dir"):
        kwargs["cache_dir"] = cfg["cache_dir"]
    try:
        return build_dataloader(**kwargs)
    except TypeError:  # pragma: no cover - older signature
        kwargs.pop("max_samples", None)
        kwargs.pop("synthetic_length", None)
        return build_dataloader(**kwargs)


def generate_samples(
    model: Any,
    coupling: Any,
    cfg: Dict[str, Any],
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    loader: Any = None,
    device: Any = None,
    generator: Any = None,
    synthetic: bool = False,
    qualitative_count: int = 8,
    progress: bool = True,
    mask_mode: str = "random",
    fixed_mask: Any = None,
) -> Dict[str, Any]:
    """Generate ``num_samples`` in-painted images and return the sampling payload."""
    import torch

    num_samples = int(num_samples or cfg.get("num_samples", 50000))
    batch_size = int(batch_size or cfg.get("batch_size", 32))
    device = device or torch.device("cpu")

    if loader is None:
        loader = build_loader(cfg, num_samples, batch_size, synthetic=synthetic)

    collected: Dict[str, List[Any]] = {"samples": [], "base": [], "ground_truth": [], "xi": [], "labels": []}
    trajectory_kept: Optional[Any] = None
    done = 0
    for batch in loader:
        x1, labels, _low = unpack_batch(batch)
        if x1 is None:
            continue
        x1 = x1.to(device)
        if labels is not None:
            labels = labels.to(device)

        want_trajectory = bool(cfg.get("save_trajectory")) and trajectory_kept is None
        xi = None
        if fixed_mask is not None:
            xi = fixed_mask.to(device)
            if xi.shape[-1] != x1.shape[-1] or xi.shape[-2] != x1.shape[-2]:
                import torch.nn.functional as F

                xi = F.interpolate(xi, size=x1.shape[-2:], mode="nearest")
            if xi.shape[0] == 1 and x1.shape[0] > 1:
                xi = xi.expand(x1.shape[0], -1, -1, -1)
        elif mask_mode != "random":
            xi = make_mask(
                x1,
                num_tiles=int(cfg.get("num_tiles", 64)),
                missing_prob=float(cfg.get("missing_prob", 0.3)),
                mode=mask_mode,
                generator=generator,
                device=device,
            )

        with torch.no_grad():
            out = sample_batch_with_mask(
                model,
                coupling,
                cfg,
                x1,
                xi=xi,
                labels=labels,
                generator=generator,
                return_trajectory=want_trajectory,
            )
        collected["samples"].append(out["sample"].detach().float().cpu())
        collected["base"].append(out["x0"].detach().float().cpu())
        collected["ground_truth"].append(x1.detach().float().cpu())
        if out.get("xi") is not None:
            collected["xi"].append(out["xi"].detach().float().cpu())
        if labels is not None:
            collected["labels"].append(labels.detach().cpu())
        if want_trajectory and out.get("trajectory") is not None:
            trajectory_kept = out["trajectory"].detach().float().cpu()

        done += int(x1.shape[0])
        if progress:
            print(f"\r[inpainting] generated {min(done, num_samples)}/{num_samples}", end="", flush=True)
        if done >= num_samples:
            break
    if progress:
        print()

    def _cat(key: str) -> Any:
        items = collected.get(key) or []
        if not items:
            return None
        return torch.cat(items, dim=0)[:num_samples]

    payload: Dict[str, Any] = {
        "samples": _cat("samples"),
        "base": _cat("base"),
        "ground_truth": _cat("ground_truth"),
        "xi": _cat("xi"),
        "labels": _cat("labels"),
        "task": "inpainting",
        "resolution": int(cfg.get("resolution", 256)),
        "method": str(cfg.get("method", "dopri5")),
        "steps": int(cfg.get("steps", 50)),
        "coefficients": str(cfg.get("coefficients", "inpainting")),
        "base_mode": str(cfg.get("base_mode", "coupled")),
        "num_tiles": int(cfg.get("num_tiles", 64)),
        "missing_prob": float(cfg.get("missing_prob", 0.3)),
        "num_samples": int(num_samples if payload_len(payload := None) is None else 0) if False else num_samples,
    }
    if trajectory_kept is not None:
        payload["trajectory"] = trajectory_kept
    count = 0 if payload["samples"] is None else int(payload["samples"].shape[0])
    payload["num_generated"] = count
    payload["qualitative"] = _make_qualitative(payload, qualitative_count)
    return payload


def payload_len(payload: Any) -> Optional[int]:  # pragma: no cover - tiny helper for clarity
    if payload is None:
        return None
    samples = payload.get("samples") if isinstance(payload, dict) else None
    return None if samples is None else int(samples.shape[0])


def _make_qualitative(payload: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    """Build base/model/ground-truth triples (Fig. 3) from the payload."""
    triples: List[Dict[str, Any]] = []
    try:
        import torch
    except Exception:  # pragma: no cover
        return triples
    samples = payload.get("samples")
    base = payload.get("base")
    truth = payload.get("ground_truth")
    xi = payload.get("xi")
    labels = payload.get("labels")
    if samples is None or base is None or truth is None:
        return triples
    n = min(int(count), int(samples.shape[0]))
    for index in range(n):
        triple: Dict[str, Any] = {
            "base": base[index : index + 1],
            "model": samples[index : index + 1],
            "ground_truth": truth[index : index + 1],
            "task": "inpainting",
        }
        if xi is not None:
            triple["xi"] = xi[index : index + 1]
        if labels is not None:
            triple["class_label"] = int(labels[index].item()) if labels[index].numel() == 1 else labels[index]
        triples.append(triple)
    return triples


def check_inpainting_consistency(payload: Dict[str, Any], atol: float = 1e-4) -> Dict[str, float]:
    """Verify the structural invariant ``xi o X_{t=1} = xi o x_1`` on observed pixels."""
    import torch

    samples = payload.get("samples")
    truth = payload.get("ground_truth")
    xi = payload.get("xi")
    if samples is None or truth is None or xi is None:
        return {}
    observed = (xi > 0.5).to(samples.dtype)
    if observed.shape[1] == 1 and samples.shape[1] > 1:
        observed = observed.expand_as(samples)
    diff = ((samples - truth).abs() * observed).reshape(samples.shape[0], -1)
    denom = observed.reshape(samples.shape[0], -1).sum(dim=1).clamp_min(1.0)
    mean_abs = float((diff.sum(dim=1) / denom).max().item())
    max_abs = float((((samples - truth).abs() * observed).amax(dim=(-1, -2, -3))).max().item())
    return {
        "observed_max_abs_error": max_abs,
        "observed_mean_abs_error": mean_abs,
        "consistent": float(max_abs <= float(atol)),
        "num_samples": float(samples.shape[0]),
    }


# --------------------------------------------------------------------------------------
# Output / evaluation
# --------------------------------------------------------------------------------------
def save_samples(payload: Dict[str, Any], outdir: str, stem: str = "samples_inpainting") -> str:
    """Persist the sampling payload (excluding figure bookkeeping) as a ``.pt`` bundle."""
    import torch

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{stem}.pt")
    to_save = {key: value for key, value in payload.items() if key not in ("qualitative",)}
    torch.save(to_save, path)
    logger.info("saved samples to %s", path)
    return path


def save_qualitative(payload: Dict[str, Any], cfg: Dict[str, Any], outdir: str) -> List[str]:
    """Write Fig. 3-style base/model/GT triples (and Fig. 5-style slices if available)."""
    os.makedirs(outdir, exist_ok=True)
    written: List[str] = []
    triples = payload.get("qualitative") or _make_qualitative(payload, int(cfg.get("qualitative_count", 8)))

    # Preferred path: reuse the shared entry-point helper when present.
    try:
        module = _require("sample")
        helper = getattr(module, "save_qualitative", None)
        if helper is not None:
            paths = helper(payload, cfg, outdir)
            if isinstance(paths, str):
                return [paths]
            if paths:
                return list(paths)
    except Exception:  # pragma: no cover - fall back to eval.qualitative directly
        pass

    try:
        TripleFigure = _require("eval.qualitative", "TripleFigure")
        save_triples_grid = _require("eval.qualitative", "save_triples_grid")
        figures = []
        for triple in triples:
            figures.append(
                TripleFigure(
                    base=triple["base"],
                    model=triple["model"],
                    ground_truth=triple["ground_truth"],
                    xi=triple.get("xi"),
                    class_label=triple.get("class_label"),
                    caption=None,
                    extra={"task": "inpainting"},
                )
            )
        if figures:
            path = save_triples_grid(
                figures,
                os.path.join(outdir, "inpainting_triples.png"),
                title="In-painting: base sample $x_0$, model sample $X_{t=1}$, ground truth",
            )
            written.append(path)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("could not write qualitative triples: %s", exc)

    trajectory = payload.get("trajectory")
    if trajectory is not None:
        try:
            make_probability_flow_figure = _require("eval.qualitative", "make_probability_flow_figure")
            path = make_probability_flow_figure(
                trajectory[:1],
                ground_truth=(payload.get("ground_truth") or torch_slice(payload["samples"]))[:1],
                mask=(payload.get("xi") or torch_slice(payload["samples"]))[:1] if payload.get("xi") is not None else None,
                path=os.path.join(outdir, "inpainting_probability_flow.png"),
                title="Probability-flow ODE slices (in-painting)",
            )
            written.append(path)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("could not write probability-flow figure: %s", exc)
    return written


def torch_slice(tensor: Any) -> Any:  # pragma: no cover - defensive helper
    return tensor


def compute_fid(payload: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[Any]:
    """Compute FID-50k for the generated samples (Table 2)."""
    try:
        evaluate_fid_50k = _require("eval.fid", "evaluate_fid_50k")
    except ImportError as exc:  # pragma: no cover
        logger.warning("FID evaluation unavailable: %s", exc)
        return None
    samples = payload.get("samples")
    if samples is None:
        return None
    return evaluate_fid_50k(
        samples,
        task="inpainting",
        batch_size=int(cfg.get("batch_size", 32)),
        num_samples=int(cfg.get("num_samples", 50000)),
        resolution=int(cfg.get("resolution", 256)),
        paper_fid=PAPER_FID_50K,
    )


def write_summary(payload: Dict[str, Any], cfg: Dict[str, Any], outdir: str, extra: Optional[Dict[str, Any]] = None) -> str:
    """Write a JSON summary of the sampling run."""
    os.makedirs(outdir, exist_ok=True)
    payload = payload or {}
    summary: Dict[str, Any] = {
        "task": "inpainting",
        "resolution": int(cfg.get("resolution", 256)),
        "coefficients": str(cfg.get("coefficients", "inpainting")),
        "sigma": float(cfg.get("sigma", 1.0)),
        "num_tiles": int(cfg.get("num_tiles", 64)),
        "missing_prob": float(cfg.get("missing_prob", 0.3)),
        "method": str(cfg.get("method", "dopri5")),
        "steps": int(cfg.get("steps", 50)),
        "base_mode": str(cfg.get("base_mode", "coupled")),
        "num_generated": int(payload.get("num_generated", 0) or 0),
        "consistency": payload.get("consistency") or {},
        "fid_50k": payload.get("fid_50k"),
        "paper_fid_50k": PAPER_FID_50K,
        "baseline_fid_50k": BASELINE_FID_50K,
    }
    if extra:
        summary.update(extra)
    path = os.path.join(outdir, "sample_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    logger.info("wrote summary to %s", path)
    return path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample from a trained in-painting stochastic-interpolant model (Section 4.1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="YAML config (e.g. configs/inpainting_256.yaml)")
    parser.add_argument("--checkpoint", default=None, help="trained U-Net checkpoint (.pt)")
    parser.add_argument("--outdir", default="samples/inpainting", help="output directory")
    parser.add_argument("--resolution", type=int, default=None, choices=[256, 512], help="ImageNet resolution")
    parser.add_argument("--split", default=None, help="dataset split used to draw x_1 (validation/train)")
    parser.add_argument("--num-samples", type=int, default=None, help="number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=None, help="sampling batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="dataloader workers")
    parser.add_argument("--method", default=None, choices=["dopri5", "dopri8", "bosh3", "euler", "rk4", "midpoint"],
                        help="ODE solver (Algorithm 2 uses forward Euler)")
    parser.add_argument("--steps", type=int, default=None, help="number of Euler/RK4 steps N")
    parser.add_argument("--atol", type=float, default=None, help="Dopri absolute tolerance")
    parser.add_argument("--rtol", type=float, default=None, help="Dopri relative tolerance")
    parser.add_argument("--sigma", type=float, default=None, help="base noise scale (Section 4.1 uses sigma = 1)")
    parser.add_argument("--num-tiles", type=int, default=None, help="number of tiles for the random mask (64)")
    parser.add_argument("--missing-prob", type=float, default=None, help="per-tile missing probability p (0.3)")
    parser.add_argument("--mask-mode", default="random", choices=["random", "center", "zeros", "ones"],
                        help="mask used for the sampled triples")
    parser.add_argument("--mask-file", default=None, help="pre-specified mask (.npy/.pt) used for all images")
    parser.add_argument("--base-mode", default=None, choices=["coupled", "independent"],
                        help="'independent' reproduces the uncoupled-interpolant baseline of Table 2")
    parser.add_argument("--no-mask-velocity", action="store_true", help="do not zero the net output on observed pixels")
    parser.add_argument("--no-project", action="store_true", help="disable per-step observed-pixel projection")
    parser.add_argument("--fid", action="store_true", help="compute FID-50k of the generated samples")
    parser.add_argument("--qualitative", action="store_true", help="write Fig. 3-style triples")
    parser.add_argument("--qualitative-count", type=int, default=None, help="number of triples to write")
    parser.add_argument("--save-trajectory", action="store_true", help="also store the ODE trajectory (Fig. 5)")
    parser.add_argument("--save-samples", action="store_true", help="save the sample tensor bundle")
    parser.add_argument("--synthetic", action="store_true", help="use synthetic random images (no ImageNet download)")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--device", default=None, help="torch device (default: auto)")
    parser.add_argument("--accelerator", default="auto", help="Lightning Fabric accelerator")
    parser.add_argument("--devices", default="auto", help="Lightning Fabric devices")
    parser.add_argument("--self-test", action="store_true", help="run the built-in smoke test and exit")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Translate CLI arguments into config overrides."""
    simple = {
        "resolution": args.resolution,
        "split": args.split,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "method": args.method,
        "steps": args.steps,
        "atol": args.atol,
        "rtol": args.rtol,
        "sigma": args.sigma,
        "num_tiles": args.num_tiles,
        "missing_prob": args.missing_prob,
        "base_mode": args.base_mode,
        "qualitative_count": args.qualitative_count,
        "seed": args.seed,
        "synthetic": True if args.synthetic else None,
        "save_trajectory": True if args.save_trajectory else None,
    }
    overrides = {key: value for key, value in simple.items() if value is not None}
    if args.no_mask_velocity:
        overrides["mask_observed"] = False
    if args.no_project:
        overrides["project_observed"] = False
    return overrides


def _setup_device(args: argparse.Namespace) -> Any:
    """Return the torch device (honouring Fabric when available)."""
    import torch

    if args.device:
        return torch.device(args.device)
    try:
        setup = _require("si.utils.distributed", "setup")
        fabric = setup(accelerator=args.accelerator, devices=args.devices, precision="32-true")
        device = getattr(fabric, "device", None)
        if device is not None:
            return torch.device(device)
    except Exception:  # pragma: no cover - Fabric is optional
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.self_test:
        return _self_test()

    import torch

    try:
        device = _setup_device(args)
        cfg = build_config(args.config, config_from_args(args))
        cfg["save_trajectory"] = bool(args.save_trajectory)
        logger.info(
            "in-painting sampling | res=%s | method=%s | tiles=%s p=%s | sigma=%s | base=%s | device=%s",
            cfg.get("resolution"), cfg.get("method"), cfg.get("num_tiles"),
            cfg.get("missing_prob"), cfg.get("sigma"), cfg.get("base_mode"), device,
        )

        coupling = build_coupling(cfg)
        model = build_model(cfg, coupling)
        load_checkpoint(model, args.checkpoint, device=device)
        model = model.to(device).eval()

        fixed_mask = load_mask_file(args.mask_file, device=device) if args.mask_file else None
        generator = make_generator(cfg.get("seed", 0), device=device)

        payload = generate_samples(
            model,
            coupling,
            cfg,
            num_samples=cfg.get("num_samples"),
            batch_size=cfg.get("batch_size"),
            device=device,
            generator=generator,
            synthetic=bool(cfg.get("synthetic", False)),
            qualitative_count=int(cfg.get("qualitative_count", 8)),
            mask_mode=args.mask_mode,
            fixed_mask=fixed_mask,
        )

        consistency = check_inpainting_consistency(payload)
        payload["consistency"] = consistency
        if consistency:
            logger.info(
                "structural check (xi o X_1 == xi o x_1): max|err| = %.3e (%s)",
                consistency.get("observed_max_abs_error", float("nan")),
                "OK" if consistency.get("consistent") else "MISMATCH",
            )

        os.makedirs(args.outdir, exist_ok=True)
        if args.qualitative:
            written = save_qualitative(payload, cfg, args.outdir)
            logger.info("wrote %d qualitative figure(s)", len(written))
        if args.save_samples:
            save_samples(payload, args.outdir)

        fid_result = None
        if args.fid:
            fid_result = compute_fid(payload, cfg)
            if fid_result is not None:
                payload["fid_50k"] = getattr(fid_result, "fid", fid_result)
                logger.info("FID-50k = %s (paper: ours %.2f | baseline %.2f)",
                            payload["fid_50k"], PAPER_FID_50K, BASELINE_FID_50K)

        write_summary(payload, cfg, args.outdir, extra={"fid_result": str(fid_result) if fid_result is not None else None})
        logger.info("done: %s sample(s) in %s", payload.get("num_generated"), args.outdir)
        return 0
    except Exception as exc:  # pragma: no cover - CLI error path
        logger.error("in-painting sampling failed: %s", exc, exc_info=True)
        return 1


# --------------------------------------------------------------------------------------
# Self test (no ImageNet required)
# --------------------------------------------------------------------------------------
def _tiny_config(resolution: int = 32) -> Dict[str, Any]:
    cfg = build_config(
        None,
        {
            "resolution": resolution,
            "channels": 3,
            "batch_size": 2,
            "num_samples": 4,
            "num_tiles": 16,
            "missing_prob": 0.3,
            "method": "euler",
            "steps": 4,
            "model": {"channels": 16, "dim_mults": [1, 2], "resnet_block_groups": 2, "num_classes": 10,
                      "attention_heads": 2, "attention_dim_head": 8},
        },
    )
    return cfg


def _self_test() -> int:
    """Smoke test: mask conventions, coupled base, ODE sampling and the structural invariant."""
    import torch

    torch.manual_seed(0)
    cfg = _tiny_config()
    device = torch.device("cpu")

    coupling = build_coupling(cfg)
    model = build_model(cfg, coupling).to(device).eval()

    # 1) Mask convention: 1 = observed, 0 = missing, shared across channels.
    xi = make_mask((2, 3, 32, 32), num_tiles=int(cfg["num_tiles"]), missing_prob=float(cfg["missing_prob"]))
    assert xi.shape == (1, 1, 32, 32), xi.shape
    assert set(torch.unique(xi).tolist()) <= {0.0, 1.0}

    # 2) Coupled base sample: observed pixels equal x1's, masked pixels are independent noise.
    x1 = torch.randn(2, 3, 32, 32)
    x0, xi_model = coupling.build_x0(x1, xi=xi, generator=torch.Generator().manual_seed(1), return_xi=True)
    observed = (xi > 0.5).float().expand_as(x1)
    assert torch.allclose(x0 * observed, x1 * observed, atol=1e-5), "observed pixels must be preserved"

    # 3) Velocity is structurally zero on observed pixels (xi o I_t = xi o x1 for all t).
    mask_fn = _as_mask_fn(coupling, xi_model)
    velocity = torch.ones_like(x1)
    if mask_fn is not None:
        masked = mask_fn(velocity)
        assert torch.allclose(masked * observed, torch.zeros_like(masked), atol=1e-5)

    # 4) ODE integration (Algorithm 2) runs end-to-end and preserves observed pixels via projection.
    out = sample_batch_with_mask(model, coupling, cfg, x1, xi=xi, generator=torch.Generator().manual_seed(2))
    assert out["sample"].shape == x1.shape, out["sample"].shape
    observed_err = ((out["sample"] - x1).abs() * observed).max().item()
    assert observed_err < 1e-3, f"observed pixels drifted: {observed_err}"

    # 5) Consistency helper and payload bookkeeping.
    payload = {
        "samples": out["sample"],
        "ground_truth": x1,
        "xi": out["xi"],
        "base": out["x0"],
        "num_generated": int(x1.shape[0]),
    }
    stats = check_inpainting_consistency(payload)
    assert stats.get("consistent") == 1.0, stats
    triples = _make_qualitative(payload, 2)
    assert len(triples) == 2 and triples[0]["base"].shape == (1, 3, 32, 32)

    # 6) Config/task defaults: in-painting uses alpha_t = t, beta_t = 1 - t (Section 4.1).
    assert cfg["coefficients"] == "inpainting", cfg["coefficients"]
    interpolant = build_interpolant(cfg)
    t0 = torch.zeros(2)
    t1 = torch.ones(2)
    a0, b0, _ = interpolant.alpha_beta_gamma(t0)
    a1, b1, _ = interpolant.alpha_beta_gamma(t1)
    assert torch.allclose(a0, torch.zeros_like(a0)) and torch.allclose(b0, torch.ones_like(b0))
    assert torch.allclose(a1, torch.ones_like(a1)) and torch.allclose(b1, torch.zeros_like(b1))
    I0 = interpolant.I_t(x0, x1, t0)
    I1 = interpolant.I_t(x0, x1, t1)
    assert torch.allclose(I0, x0, atol=1e-5) and torch.allclose(I1, x1, atol=1e-5)

    # 7) Independent (uncoupled) baseline path produces a pure-noise base sample.
    payload_base = sample_batch_with_mask(
        model, coupling, {**cfg, "base_mode": "independent", "mask_observed": False, "project_observed": False},
        x1, generator=torch.Generator().manual_seed(3),
    )
    assert payload_base["x0"].shape == x1.shape

    print("[sample_inpainting] self-test OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
