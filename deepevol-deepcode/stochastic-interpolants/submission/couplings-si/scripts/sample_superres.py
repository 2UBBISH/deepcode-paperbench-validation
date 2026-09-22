"""Super-resolution sampling entry point (paper Section 4.2, Algorithm 2).

This script produces super-resolved ImageNet images from a trained stochastic
interpolant with the *data-dependent* super-resolution coupling

    x_0 = U(D(x_1)) + sigma * zeta,      zeta ~ N(0, Id),   sigma > 0
    xi  = U(D(x_1))                      (appended to the model input channels)

where ``D`` and ``U`` are the image down/upsampling operators from Section 4.2.
Because ``rho_0(x_0 | x_1, xi)`` is concentrated near the low-resolution
observation, the probability-flow ODE (Eq. 8)

    dX_t = b_t(X_t, xi) dt,     X_{t=0} = x_0,

is integrated from ``t = 0`` to ``t = 1`` (Dopri by default, or the forward
Euler scheme of Algorithm 2) to produce a sample ``X_{t=1}`` at the high
resolution.  The conditioning ``xi`` is re-appended at *every* integration
step, and the ImageNet class label is fed through the embedding path.

Reported targets (Table 3, FID-50k, 64x64 -> 256x256)
-----------------------------------------------------
    Dependent Coupling (Ours)    Train 2.13   Valid 2.05

The baseline FIDs (Improved DDPM, SR3, ADM, Cascaded Diffusion, I^2SB) are
taken from the reported literature and are *not* re-run.

Usage
-----
    python scripts/sample_superres.py --config configs/superres_64_256.yaml \
        --checkpoint runs/superres_64_256/model.pt --outdir samples/sr_64_256 \
        --method dopri5 --fid --qualitative

    # smoke test without ImageNet / checkpoint
    python scripts/sample_superres.py --self-test
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Repo-root import shim so the script runs standalone.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

__all__ = [
    "DEFAULT_SUPERRES_CONFIG",
    "PAPER_FIDS",
    "build_config",
    "resolve_task_settings",
    "build_coupling",
    "build_interpolant",
    "build_model",
    "load_checkpoint",
    "make_low_res",
    "sample_batch_superres",
    "generate_samples",
    "superres_statistics",
    "save_samples",
    "save_qualitative",
    "compute_fid",
    "write_summary",
    "parse_args",
    "config_from_args",
    "main",
]

LOGGER = logging.getLogger("sample_superres")

# ---------------------------------------------------------------------------
# Paper constants
# ---------------------------------------------------------------------------

#: Table 3, FID-50k for the 64x64 -> 256x256 task.
PAPER_FIDS: Dict[str, Dict[str, float]] = {
    "64_256": {"train": 2.13, "valid": 2.05},
    "256_512": {},
}

#: Baselines that must NOT be re-run (values copied from prior work).
REPORTED_ONLY_BASELINES: Dict[str, Dict[str, float]] = {
    "64_256": {
        "Improved DDPM (Nichol & Dhariwal, 2021)": 12.26,
        "SR3 (Saharia et al., 2022)": 11.30,
        "ADM (Dhariwal & Nichol, 2021)": 7.49,
        "Cascaded Diffusion (Ho et al., 2022a)": 4.88,
        "I^2SB (Liu et al., 2023a)": 2.70,
    },
}

ADAPTIVE_METHODS = ("dopri5", "dopri8", "bosh3", "fehlberg2", "adaptive_heun")
FIXED_METHODS = ("euler", "midpoint", "rk4", "explicit_adams", "implicit_adams")

#: Task defaults for super-resolution (Section 4.2 + Appendix B + addendum).
DEFAULT_SUPERRES_CONFIG: Dict[str, Any] = {
    "task": "superres",
    # -- coupling (Section 4.2) ------------------------------------------
    "sigma": 0.05,              # sigma > 0 smooths the base density
    "coefficients": "gamma0",   # alpha_t = 1 - t, beta_t = t, gamma_t = 0
    "low_res": 64,              # W_low = H_low
    "down_mode": "area",        # D
    "up_mode": "bilinear",      # U
    "antialias": True,
    # -- data -------------------------------------------------------------
    "resolution": 256,
    "batch_size": 32,
    "num_workers": 4,
    "split": "train",
    "low_res_view": "native",   # "native" -> D(x1); "upsampled" -> U(D(x1))
    "synthetic": False,
    # -- model (Appendix B) ----------------------------------------------
    "channels": 256,
    "dim_mults": (1, 1, 2, 3, 4),
    "resnet_block_groups": 8,
    "learned_sinusoidal_cond": True,
    "learned_sinusoidal_dim": 32,
    "attention_dim_head": 64,
    "attention_heads": 4,
    "random_fourier_features": False,
    "num_classes": 1000,
    "conditioning_channels": 3,  # xi = U(D(x1)) appended as extra channels
    "in_channels": 3,
    "mask_observed": False,      # structural masking is in-painting only
    # -- sampling ---------------------------------------------------------
    "method": "dopri5",
    "steps": 50,
    "atol": 1e-5,
    "rtol": 1e-5,
    "num_samples": 50_000,
    "qualitative_count": 8,
    "project_observed": False,   # no exact fixed-pixel invariant for SR
}

_CLI_TO_CONFIG = {
    "task": "task",
    "resolution": "resolution",
    "low_res": "low_res",
    "sigma": "sigma",
    "coefficients": "coefficients",
    "down_mode": "down_mode",
    "up_mode": "up_mode",
    "method": "method",
    "steps": "steps",
    "batch_size": "batch_size",
    "num_samples": "num_samples",
    "channels": "channels",
    "num_classes": "num_classes",
    "split": "split",
    "num_workers": "num_workers",
    "atol": "atol",
    "rtol": "rtol",
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def deep_update(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _require(module_name: str, attr: Optional[str] = None):
    """Import ``attr`` from ``module_name`` lazily, with a clear error."""
    import importlib

    module = importlib.import_module(module_name)
    if attr is None:
        return module
    try:
        return getattr(module, attr)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ImportError(f"{module_name} has no attribute {attr!r}") from exc


def _read_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        try:
            from si.utils.config import load_config  # type: ignore

            cfg = load_config(path)
            return dict(cfg) if cfg is not None else {}
        except Exception:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)


def _split_payload(payload: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    return {k: payload[k] for k in keys if k in payload and payload[k] is not None}


def make_generator(seed: Optional[int], device: Optional[torch.device] = None):
    if seed is None:
        return None
    generator = torch.Generator(device=device if device is not None else "cpu")
    generator.manual_seed(int(seed))
    return generator


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def resolve_task_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Force the super-resolution conventions of Section 4.2."""
    cfg = dict(cfg)
    cfg["task"] = "superres"
    cfg.setdefault("coefficients", "gamma0")
    cfg.setdefault("sigma", 0.05)
    cfg.setdefault("down_mode", "area")
    cfg.setdefault("up_mode", "bilinear")
    cfg["mask_observed"] = False
    cfg.setdefault("in_channels", 3)
    cfg["conditioning_channels"] = int(cfg.get("conditioning_channels") or cfg["in_channels"])
    if cfg.get("low_res") is None:
        cfg["low_res"] = 256 if int(cfg.get("resolution", 256)) >= 512 else 64
    cfg.setdefault("low_res_view", "native")
    return cfg


def build_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compose DEFAULT -> YAML -> overrides and apply SR task defaults."""
    cfg = copy.deepcopy(DEFAULT_SUPERRES_CONFIG)
    if path:
        cfg = deep_update(cfg, _read_yaml(path))
    if overrides:
        cfg = deep_update(cfg, overrides)
    return resolve_task_settings(cfg)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def build_coupling(cfg: Dict[str, Any], device: Optional[torch.device] = None):
    """Instantiate :class:`si.couplings.SuperresCoupling` from config."""
    try:
        from si.couplings import SuperresCoupling  # type: ignore
    except Exception:
        SuperresCoupling = _require("si.couplings.superres", "SuperresCoupling")
    return SuperresCoupling(
        low_res=int(cfg["low_res"]),
        sigma=float(cfg.get("sigma", 0.05)),
        down_mode=cfg.get("down_mode", "area"),
        up_mode=cfg.get("up_mode", "bilinear"),
        antialias=bool(cfg.get("antialias", True)),
        coefficients=cfg.get("coefficients", "gamma0"),
        requires_conditioning=True,
    )


def build_interpolant(cfg: Dict[str, Any]):
    try:
        from si.interpolants import Interpolant, get_coefficients  # type: ignore
    except Exception:
        Interpolant = _require("si.interpolants.interpolant", "Interpolant")
        get_coefficients = _require("si.interpolants.coefficients", "get_coefficients")
    return Interpolant(get_coefficients(cfg.get("coefficients", "gamma0")))


def _model_kwargs(cfg: Dict[str, Any], coupling=None) -> Dict[str, Any]:
    in_channels = int(cfg.get("in_channels", 3))
    conditioning_channels = int(cfg.get("conditioning_channels", in_channels))
    return dict(
        in_channels=in_channels,
        channels=int(cfg.get("channels", 256)),
        dim_mults=tuple(cfg.get("dim_mults", (1, 1, 2, 3, 4))),
        resnet_block_groups=int(cfg.get("resnet_block_groups", 8)),
        num_classes=cfg.get("num_classes", 1000),
        class_dropout_prob=float(cfg.get("class_dropout_prob", 0.1)),
        conditioning_channels=conditioning_channels,
        learned_sinusoidal_cond=bool(cfg.get("learned_sinusoidal_cond", True)),
        learned_sinusoidal_dim=int(cfg.get("learned_sinusoidal_dim", 32)),
        attention_dim_head=int(cfg.get("attention_dim_head", 64)),
        attention_heads=int(cfg.get("attention_heads", 4)),
        random_fourier_features=bool(cfg.get("random_fourier_features", False)),
        mask_observed=False,
        image_size=int(cfg.get("resolution", 256)),
    )


def build_model(cfg: Dict[str, Any], coupling=None) -> torch.nn.Module:
    """Build the Appendix-B velocity U-Net (``2*C`` input channels for SR)."""
    kwargs = _model_kwargs(cfg, coupling)
    try:
        from si.models import unet_from_config  # type: ignore

        return unet_from_config(kwargs)
    except Exception:
        VelocityUNet = _require("si.models.unet", "VelocityUNet")
        return VelocityUNet(**kwargs)


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    device: Optional[torch.device] = None,
    use_ema: bool = True,
) -> Dict[str, Any]:
    """Load model weights, preferring EMA weights when present."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path!r}")
    map_location = device or "cpu"
    if os.path.isdir(path):  # pragma: no cover - dir checkpoints
        candidate = os.path.join(path, "model.pt")
        if os.path.isfile(candidate):
            path = candidate

    try:
        from train import Trainer  # type: ignore

        return Trainer.load_into(model, path, map_location=map_location, use_ema=use_ema)
    except Exception:
        pass

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        model.load_state_dict(payload, strict=False)
        return {"loaded": "state_dict"}
    for key in (("ema", "shadow") if use_ema else (), ("model",), ("state_dict",), ("weights",)):
        for name in key:
            if name in payload and isinstance(payload[name], dict):
                state = payload[name]
                state = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state.items()}
                model.load_state_dict(state, strict=False)
                LOGGER.info("loaded weights from payload[%r]", name)
                return payload
    model.load_state_dict(payload, strict=False)
    return {"loaded": "payload"}


# ---------------------------------------------------------------------------
# super-resolution helpers
# ---------------------------------------------------------------------------
def make_low_res(
    x1: torch.Tensor,
    coupling,
    cfg: Optional[Dict[str, Any]] = None,
    low: Optional[torch.Tensor] = None,
    upsampled: bool = True,
) -> torch.Tensor:
    """Return ``U(D(x1))`` (default) or ``D(x1)``.

    ``xi = U(D(x_1))`` is the conditioning tensor appended to the model input
    channels at every timestep (Section 4.2).
    """
    if low is None:
        low = coupling.D(x1)
    if upsampled:
        return coupling.U(low, high_res=x1.shape[-2:])
    return low


def draw_base(
    coupling,
    x1: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    xi: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the coupled base ``x_0 = U(D(x_1)) + sigma*zeta`` and ``xi``."""
    if xi is None:
        xi = coupling.conditioning(x1)
    x0, xi_out = coupling.build_x0(x1, xi=xi, generator=generator, return_xi=True)
    return x0, xi_out, (x0 - coupling.m(x1, xi_out))


def sample_batch_superres(
    model: torch.nn.Module,
    coupling,
    cfg: Dict[str, Any],
    x1: torch.Tensor,
    low: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    return_trajectory: bool = False,
    x0: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Integrate the probability-flow ODE for one super-resolution batch.

    ``xi = U(D(x_1))`` is closed over and re-appended by the sampler at every
    integration step (the model input has ``2*C`` channels).
    """
    try:
        from si.samplers import probability_flow_ode  # type: ignore
    except Exception:
        probability_flow_ode = _require("si.samplers.ode", "probability_flow_ode")

    xi = make_low_res(x1, coupling, cfg, low=low, upsampled=True)
    zeta = None
    if x0 is None:
        if generator is None:
            zeta = torch.randn_like(x1)
        else:
            zeta = torch.randn(x1.shape, generator=generator, device=x1.device, dtype=x1.dtype)
        x0 = coupling.m(x1, xi=xi) + float(coupling.sigma) * zeta

    method = str(cfg.get("method", "dopri5"))
    kwargs: Dict[str, Any] = {}
    if method in ADAPTIVE_METHODS:
        kwargs["atol"] = float(cfg.get("atol", 1e-5))
        kwargs["rtol"] = float(cfg.get("rtol", 1e-5))
        if return_trajectory:
            kwargs["trajectory_steps"] = int(cfg.get("steps", 50))
    else:
        kwargs["steps"] = int(cfg.get("steps", 50))

    out = probability_flow_ode(
        model,
        x0,
        xi=xi,
        y=labels,
        method=method,
        return_trajectory=bool(return_trajectory),
        **kwargs,
    )
    if return_trajectory:
        sample, trajectory = out
    else:
        sample, trajectory = out, None

    result: Dict[str, Any] = {"sample": sample, "x0": x0, "x1": x1, "xi": xi}
    if zeta is not None:
        result["zeta"] = zeta
    if labels is not None:
        result["label"] = labels
    if low is not None:
        result["low"] = low
    if trajectory is not None:
        result["trajectory"] = trajectory
    return result


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract ``(x1, labels, low_res)`` from a dict/tuple batch."""
    if isinstance(batch, dict):
        x1 = batch.get("x1", batch.get("image", batch.get("images")))
        labels = batch.get("label", batch.get("labels", batch.get("y")))
        low = batch.get("low", batch.get("low_res", batch.get("low_resolution")))
        return x1, labels, low
    if isinstance(batch, (tuple, list)):
        if len(batch) == 3:
            return batch[0], batch[2], batch[1]
        if len(batch) == 2:
            return batch[0], batch[1], None
        return batch[0], None, None
    return batch, None, None


def build_loader(
    cfg: Dict[str, Any],
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    fabric=None,
    split: Optional[str] = None,
    synthetic: Optional[bool] = None,
):
    build_dataloader = _require("si.data", "build_dataloader")
    low_res = int(cfg["low_res"])
    return build_dataloader(
        split=split or cfg.get("split", "train"),
        batch_size=int(batch_size or cfg.get("batch_size", 32)),
        resolution=int(cfg.get("resolution", 256)),
        low_resolution=low_res,
        num_workers=int(cfg.get("num_workers", 4)),
        synthetic=bool(cfg.get("synthetic", False) if synthetic is None else synthetic),
        max_samples=num_samples,
        return_dict=True,
    )


# ---------------------------------------------------------------------------
# generation / statistics / output
# ---------------------------------------------------------------------------
def generate_samples(
    model: torch.nn.Module,
    coupling,
    cfg: Dict[str, Any],
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    loader=None,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
    synthetic: bool = False,
    qualitative_count: int = 8,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run the Sampler (Algorithm 2) over a dataloader and aggregate results."""
    num_samples = int(num_samples or cfg.get("num_samples", 50_000))
    batch_size = int(batch_size or cfg.get("batch_size", 32))
    if loader is None:
        loader = build_loader(cfg, num_samples, batch_size, split="train", synthetic=synthetic)

    model.eval()
    collected: Dict[str, List[torch.Tensor]] = {
        "samples": [],
        "base": [],
        "ground_truth": [],
        "xi": [],
        "low": [],
        "labels": [],
        "trajectory": [],
    }
    done = 0
    iterator = loader
    if progress:
        try:
            from tqdm import tqdm  # type: ignore

            iterator = tqdm(loader, desc="sampling (superres)", leave=False)
        except Exception:  # pragma: no cover
            iterator = loader

    with torch.no_grad():
        for batch in iterator:
            x1, labels, low = unpack_batch(batch)
            if x1 is None:
                continue
            if device is not None:
                x1 = x1.to(device)
                if labels is not None:
                    labels = labels.to(device)
                if low is not None:
                    low = low.to(device)
            want_traj = bool(cfg.get("save_trajectory", False)) and done < qualitative_count
            out = sample_batch_superres(
                model,
                coupling,
                cfg,
                x1,
                low=low,
                labels=labels,
                generator=generator,
                return_trajectory=want_traj,
            )
            take = min(int(x1.shape[0]), num_samples - done)
            collected["samples"].append(out["sample"][:take].detach().cpu())
            collected["base"].append(out["x0"][:take].detach().cpu())
            collected["ground_truth"].append(out["x1"][:take].detach().cpu())
            collected["xi"].append(out["xi"][:take].detach().cpu())
            if low is not None:
                collected["low"].append(low[:take].detach().cpu())
            elif "low" in out:
                collected["low"].append(out["low"][:take].detach().cpu())
            if labels is not None:
                collected["labels"].append(labels[:take].detach().cpu())
            if "trajectory" in out:
                collected["trajectory"].append(out["trajectory"].detach().cpu())
            done += take
            if done >= num_samples:
                break

    payload: Dict[str, Any] = {"num_generated": done}
    for key, chunks in collected.items():
        if chunks:
            payload[key] = torch.cat(chunks, dim=0)

    payload["qualitative"] = _make_qualitative(payload, qualitative_count)
    return payload


def _make_qualitative(payload: Dict[str, Any], count: int) -> List[Any]:
    """Build Fig. 4 style (low-res, model sample, high-res) triples."""
    make_triples = None
    try:
        from eval.qualitative import make_triples  # type: ignore
    except Exception:
        try:
            make_triples = _require("eval.qualitative", "make_triples")
        except Exception:
            return []

    samples = payload.get("samples")
    gt = payload.get("ground_truth")
    if samples is None or gt is None:
        return []
    left = payload.get("low")
    if left is None:
        left = payload.get("xi")
    if left is None:
        left = samples  # degenerate fallback: show the sample as its own input
    n = min(int(count), int(samples.shape[0]))
    labels = payload.get("labels")
    label_list = None
    if labels is not None:
        label_list = [int(v) for v in labels[:n].reshape(-1).tolist()]
    xi = payload.get("xi")
    try:
        return make_triples(
            left[:n],
            samples[:n],
            gt[:n],
            xi=None if xi is None else xi[:n],
            class_labels=label_list,
            task="superres",
            low_res=left[:n],
        )
    except TypeError:
        return make_triples(left[:n], samples[:n], gt[:n])


def superres_statistics(payload: Dict[str, Any], coupling=None) -> Dict[str, float]:
    """Diagnostics for the super-resolution run.

    There is no exact fixed-pixel invariant for super-resolution (unlike
    in-painting).  We instead report how close the *low-resolution* content of
    the model sample is to that of the ground truth, i.e.
    ``D(X_{t=1})`` vs ``D(x_1)``, which should be small when the time-zero
    coupling places each base pixel near its target.
    """
    stats: Dict[str, float] = {}
    samples = payload.get("samples")
    gt = payload.get("ground_truth")
    if samples is None or gt is None:
        return stats
    stats["num_samples"] = float(samples.shape[0])
    diff = (samples - gt).abs()
    stats["pixel_mae"] = float(diff.mean().item())
    stats["pixel_rmse"] = float((samples - gt).pow(2).mean().sqrt().item())
    if coupling is not None:
        try:
            d_s = coupling.D(samples)
            d_g = coupling.D(gt)
        except Exception:  # pragma: no cover - defensive
            d_s = d_g = None
        if d_s is not None and d_g is not None:
            low_diff = (d_s - d_g).abs()
            stats["low_res_mae"] = float(low_diff.mean().item())
            stats["low_res_max_abs_error"] = float(low_diff.amax().item())
    return stats


def save_samples(payload: Dict[str, Any], outdir: str, stem: str = "samples_superres") -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{stem}.pt")
    save_payload = {k: v for k, v in payload.items() if k != "qualitative"}
    torch.save(save_payload, path)
    LOGGER.info("saved samples -> %s", path)
    return path


def save_qualitative(payload: Dict[str, Any], cfg: Dict[str, Any], outdir: str) -> List[str]:
    """Write Fig. 4 triples grid (+ optional probability-flow figure, Fig. 5)."""
    os.makedirs(outdir, exist_ok=True)
    written: List[str] = []

    # Preferred: reuse the shared helper from sample.py when available.
    try:
        from sample import save_qualitative as _shared  # type: ignore

        outs = _shared(payload, cfg, outdir)
        if isinstance(outs, str):
            return [outs]
        if outs:
            return list(outs)
    except Exception:
        pass

    triples = payload.get("qualitative")
    if triples:
        try:
            save_triples_grid = _require("eval.qualitative", "save_triples_grid")
            path = os.path.join(outdir, "triples_superres.png")
            save_triples_grid(
                triples,
                path,
                column_titles=(
                    "Low resolution",
                    "Model sample $X_{t=1}$",
                    "Ground truth",
                ),
                title=f"Super-resolution {cfg.get('low_res')}x{cfg.get('low_res')}"
                f" -> {cfg.get('resolution')}x{cfg.get('resolution')}",
            )
            written.append(path)
        except Exception as exc:  # pragma: no cover - optional matplotlib
            LOGGER.warning("could not write triples figure: %s", exc)

    trajectory = payload.get("trajectory")
    if trajectory is not None:
        try:
            make_probability_flow_figure = _require(
                "eval.qualitative", "make_probability_flow_figure"
            )
            path = os.path.join(outdir, "probability_flow_superres.png")
            gt = payload.get("ground_truth")
            make_probability_flow_figure(
                trajectory[0],
                ground_truth=None if gt is None else gt[0],
                path=path,
                title="Probability-flow ODE (super-resolution)",
            )
            written.append(path)
        except Exception as exc:  # pragma: no cover - optional matplotlib
            LOGGER.warning("could not write probability-flow figure: %s", exc)
    return written


def compute_fid(payload: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[Any]:
    try:
        from eval.fid import evaluate_fid_50k  # type: ignore
    except Exception:
        try:
            evaluate_fid_50k = _require("eval.fid", "evaluate_fid_50k")
        except Exception:
            LOGGER.warning("eval.fid unavailable; skipping FID")
            return None
    key = f"{int(cfg.get('low_res', 64))}_{int(cfg.get('resolution', 256))}"
    paper = None
    if os.environ.get("SI_FID_SPLIT", ""):
        paper = PAPER_FIDS.get(key, {}).get(os.environ["SI_FID_SPLIT"])
    task = f"superres_{key}"
    try:
        return evaluate_fid_50k(
            payload["samples"],
            task=task,
            num_samples=int(cfg.get("num_samples", 50_000)),
            paper_fid=paper,
            resolution=int(cfg.get("resolution", 256)),
        )
    except TypeError:
        return evaluate_fid_50k(payload["samples"])


def write_summary(payload: Dict[str, Any], cfg: Dict[str, Any], outdir: str, extra: Optional[Dict[str, Any]] = None) -> str:
    os.makedirs(outdir, exist_ok=True)
    key = f"{int(cfg.get('low_res', 64))}_{int(cfg.get('resolution', 256))}"
    summary: Dict[str, Any] = {
        "task": "superres",
        "low_res": int(cfg.get("low_res", 64)),
        "resolution": int(cfg.get("resolution", 256)),
        "sigma": float(cfg.get("sigma", 0.05)),
        "coefficients": cfg.get("coefficients", "gamma0"),
        "down_mode": cfg.get("down_mode", "area"),
        "up_mode": cfg.get("up_mode", "bilinear"),
        "method": cfg.get("method", "dopri5"),
        "steps": cfg.get("steps", 50),
        "num_generated": int(payload.get("num_generated", 0)),
        "paper_fid_50k": PAPER_FIDS.get(key, {}),
        "reported_only_baselines": REPORTED_ONLY_BASELINES.get(key, {}),
    }
    if extra:
        summary.update(extra)
    path = os.path.join(outdir, "sample_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("wrote summary -> %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample ImageNet super-resolution (Section 4.2, Algorithm 2)"
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, default=None, help="model checkpoint (.pt)")
    parser.add_argument("--outdir", type=str, default="samples/superres")
    parser.add_argument("--task", type=str, default="superres")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--low-res", type=int, default=None, dest="low_res")
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--coefficients", type=str, default=None)
    parser.add_argument("--down-mode", type=str, default=None, dest="down_mode")
    parser.add_argument("--up-mode", type=str, default=None, dest="up_mode")
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--num-samples", type=int, default=None, dest="num_samples")
    parser.add_argument("--num-workers", type=int, default=None, dest="num_workers")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--fid", action="store_true", help="compute FID-50k")
    parser.add_argument("--qualitative", action="store_true", help="write Fig.4 triples")
    parser.add_argument(
        "--save-trajectory", action="store_true", dest="save_trajectory",
        help="store ODE trajectory snapshots for probability-flow figures",
    )
    parser.add_argument("--synthetic", action="store_true", help="use synthetic data (no download)")
    parser.add_argument("--no-ema", action="store_true", dest="no_ema")
    parser.add_argument("--self-test", action="store_true", dest="self_test")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for arg_name, cfg_key in _CLI_TO_CONFIG.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[cfg_key] = value
    if getattr(args, "save_trajectory", False):
        overrides["save_trajectory"] = True
    if getattr(args, "synthetic", False):
        overrides["synthetic"] = True
    return overrides


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.self_test:
        return _self_test()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    cfg = build_config(args.config, config_from_args(args))
    LOGGER.info(
        "super-resolution %dx%d -> %dx%d | sigma=%s coeff=%s method=%s",
        cfg["low_res"], cfg["low_res"], cfg["resolution"], cfg["resolution"],
        cfg.get("sigma"), cfg.get("coefficients"), cfg.get("method"),
    )

    coupling = build_coupling(cfg, device=device)
    model = build_model(cfg, coupling)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device=device, use_ema=not args.no_ema)
    else:
        LOGGER.warning("no checkpoint given: sampling with randomly-initialised weights")
    model.to(device)
    model.eval()

    generator = make_generator(args.seed, device=device)
    payload = generate_samples(
        model,
        coupling,
        cfg,
        num_samples=int(cfg.get("num_samples", 50_000)),
        batch_size=int(cfg.get("batch_size", 32)),
        device=device,
        generator=generator,
        synthetic=bool(cfg.get("synthetic", False)),
        qualitative_count=int(cfg.get("qualitative_count", 8)),
    )

    extra: Dict[str, Any] = {"statistics": superres_statistics(payload, coupling)}
    LOGGER.info("statistics: %s", extra["statistics"])

    written = save_samples(payload, args.outdir)
    if args.qualitative:
        extra["qualitative_paths"] = save_qualitative(payload, cfg, args.outdir)
    extra["samples_path"] = written

    if args.fid:
        result = compute_fid(payload, cfg)
        if result is not None:
            extra["fid"] = result.as_dict() if hasattr(result, "as_dict") else float(result)
            LOGGER.info("FID: %s", extra["fid"])

    write_summary(payload, cfg, args.outdir, extra=extra)
    return 0


# ---------------------------------------------------------------------------
# self test (no ImageNet, no checkpoint, no torchdiffeq required)
# ---------------------------------------------------------------------------
class _TinyVelocity(torch.nn.Module):
    """Tiny velocity net with the SR signature: ``f(x, t, xi=..., y=...)``."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, t, xi=None, y=None, **kwargs):
        target = xi if xi is not None else torch.zeros_like(x)
        return self.scale * (target - x)


def _self_test() -> int:
    torch.manual_seed(0)
    cfg = build_config(None, {"synthetic": True, "resolution": 64, "low_res": 16, "batch_size": 2, "num_samples": 2})
    coupling = build_coupling(cfg)
    x1 = torch.randn(2, 3, 64, 64)

    # D/U shapes: D maps to (W_low, H_low), U back to (W, H).
    low = coupling.D(x1)
    assert low.shape[-2:] == (16, 16), low.shape
    xi = coupling.U(low, high_res=(64, 64))
    assert xi.shape == x1.shape, xi.shape
    print(f"[ok] D(x1) -> {tuple(low.shape)}, U(D(x1)) -> {tuple(xi.shape)}")

    # Coupled base: x0 = U(D(x1)) + sigma*zeta with sigma = 0.05 > 0.
    gen = torch.Generator().manual_seed(1)
    x0, xi_out = coupling.build_x0(x1, xi=xi, generator=gen, return_xi=True)
    assert x0.shape == x1.shape
    resid = (x0 - xi_out).std().item()
    assert 0.02 < resid < 0.12, resid
    print(f"[ok] x0 = U(D(x1)) + sigma*zeta (empirical sigma={resid:.4f}, cfg sigma={cfg['sigma']})")
    assert float(coupling.sigma) > 0.0
    print("[ok] sigma > 0 keeps the base density off the low-dimensional manifold")

    # Interpolant boundary conditions for the gamma0 preset.
    interp = build_interpolant(cfg)
    z = torch.randn_like(x1)
    t0 = torch.zeros(2)
    t1 = torch.ones(2)
    assert torch.allclose(interp.I_t(x0, x1, t0, z=z), x0, atol=1e-5)
    assert torch.allclose(interp.I_t(x0, x1, t1, z=z), x1, atol=1e-5)
    print("[ok] gamma0 preset: I_0 = x0, I_1 = x1")

    # End-to-end ODE with a tiny model: xi must reach the net at every step.
    model = _TinyVelocity()
    out = sample_batch_superres(
        model, coupling, cfg, x1, labels=torch.zeros(2, dtype=torch.long), generator=gen
    )
    assert out["sample"].shape == x1.shape
    err = (out["sample"] - x1).abs().mean().item()
    assert err < 1.0, err
    print(f"[ok] probability-flow ODE sample shape {tuple(out['sample'].shape)} (|X_1 - x_1| = {err:.4f})")

    stats = superres_statistics({"samples": out["sample"], "ground_truth": x1}, coupling)
    assert stats["pixel_mae"] >= 0.0
    print(f"[ok] statistics: {stats}")

    print("sample_superres self-test passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
