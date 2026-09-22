#!/usr/bin/env python3
"""Training entry point: Algorithm 1 of *Stochastic Interpolants with
Data-Dependent Couplings*.

This script implements the simulation-free training loop (Algorithm 1, page 6 of
the paper):

    Input: Interpolant coefficients alpha_t, beta_t; velocity model b_hat; batch size n_b.
    repeat
        for i = 1, ..., n_b:
            Draw x1^i ~ rho1(x1), zeta_i ~ N(0, Id), t_i ~ U(0, 1).
            Compute x0^i = m(x1^i) + sigma * zeta^i.
            Compute I_{t_i} = alpha_{t_i} x0^i + beta_{t_i} x1^i.
        Compute empirical loss
            L_hat_b(b_hat) = n_b^{-1} sum_i [ |b_hat_{t_i}(I_{t_i})|^2
                                              - 2 I_dot_{t_i} . b_hat_{t_i}(I_{t_i}) ]
        Take gradient step on L_hat_b(b_hat) to update b_hat.
    until converged
    Return: Velocity b_hat.

with the optimization settings from Appendix B / the addendum:

    * Adam optimizer, learning rate 2e-4
    * StepLR scheduler, gamma = 0.99 every N = 1000 steps
    * no weight decay
    * gradient-norm clipping at 10,000 (norm of the whole parameter vector, PyTorch default)
    * batch size 32, 200,000 gradient steps
    * PyTorch + Lightning Fabric for parallelism

and, for the U-Net velocity model, the Appendix-B hyperparameters
(dim_mults (1,1,2,3,4), channels 256, resnet-block groups 8,
learned_sinusoidal_cond True, learned_sinusoidal_dim 32, attention_dim_head 64,
attention_heads 4, random_fourier_features False).  Image-shaped conditioning is
appended to x_t at each time step: the missingness mask for in-painting, the
upsampled low-resolution image for super-resolution.

Usage
-----
    python train.py --config configs/inpainting_256.yaml
    python train.py --task inpainting --resolution 256 --steps 200000 --batch-size 32
    python train.py --task superres --resolution 256 --low-resolution 64 --synthetic \
                    --steps 200 --batch-size 4      # CPU smoke test

All experiments use batch size 32 and 200,000 gradient steps (addendum).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

# Make `si` importable when this file is run from the repository root.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from si.couplings import coupling_from_config  # noqa: E402  (defined below if missing)
from si.utils import distributed as dist_utils  # noqa: E402

logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Fallback config plumbing
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on the optional pyyaml install
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def _read_yaml(path: str) -> Dict[str, Any]:
    """Read a YAML config, falling back to the `si.utils.config` loader."""
    try:
        from si.utils import config as config_mod  # local import (lazy)

        loader = getattr(config_mod, "load_config", None)
        if callable(loader):
            loaded = loader(path)
            if hasattr(loaded, "to_dict"):
                return dict(loaded.to_dict())
            if isinstance(loaded, dict):
                return dict(loaded)
    except Exception:  # pragma: no cover - loader is optional
        pass
    if yaml is None:  # pragma: no cover
        raise RuntimeError(
            "pyyaml is required to read configuration files "
            "(pip install pyyaml) or pass flags on the command line."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(payload: Dict[str, Any], path: str) -> str:
    try:
        from si.utils import config as config_mod  # local import (lazy)

        saver = getattr(config_mod, "save_config", None)
        if callable(saver):
            saver(payload, path)
            return path
    except Exception:  # pragma: no cover
        pass
    if yaml is None:  # pragma: no cover
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        return path
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


# ---------------------------------------------------------------------------
# Default configuration (Appendix B + addendum values)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "task": "inpainting",               # "inpainting" | "superres"
    "seed": 0,
    "out_dir": "runs",
    "run_name": "si",
    # ---- data ----------------------------------------------------------
    "resolution": 256,                  # 256 or 512
    "low_resolution": None,             # 64 (SR 64->256) or 256 (SR 256->512)
    "data_root": None,
    "cache_dir": None,
    "synthetic": False,
    "streaming": False,
    "num_workers": 4,
    "train_split": "train",
    "valid_split": "validation",
    "num_classes": 1000,
    "class_dropout_prob": 0.1,
    # ---- optimization --------------------------------------------------
    "batch_size": 32,                   # addendum: all experiments
    "steps": 200000,                    # addendum: all experiments
    "lr": 2e-4,                         # Appendix B: Adam, 2e-4
    "betas": (0.9, 0.999),              # PyTorch defaults (paper silent)
    "weight_decay": 0.0,                # Appendix B: no weight decay
    "scheduler": "step_lr",
    "scheduler_gamma": 0.99,            # Appendix B: x0.99
    "scheduler_step_size": 1000,        # Appendix B: every 1000 steps
    "grad_clip": 10000.0,               # Appendix B: clip at 10,000
    "precision": "32-true",
    "ema_decay": 0.9999,
    "use_ema": True,
    # ---- model (Appendix B) --------------------------------------------
    "model": "unet",
    "channels": 256,
    "dim_mults": (1, 1, 2, 3, 4),
    "resnet_block_groups": 8,
    "learned_sinusoidal_cond": True,
    "learned_sinusoidal_dim": 32,
    "attention_dim_head": 64,
    "attention_heads": 4,
    "random_fourier_features": False,
    # ---- interpolant / coupling ----------------------------------------
    "coefficients": None,               # task default: inpainting | gamma0 (SR)
    "sigma": None,                      # coupling default
    "num_tiles": 64,                    # in-painting mask: 64 tiles
    "missing_prob": 0.3,                # in-painting mask: p = 0.3
    "down_mode": "area",
    "up_mode": "bilinear",
    "grad_accum_steps": 1,
    # ---- sampling / eval -------------------------------------------------
    "sample_every": 0,                  # >0: run the ODE sampler during training
    "sample_steps": 50,
    "sample_batches": 1,
    "sample_method": "dopri5",
    "eval_every": 0,                    # >0: compute FID (needs torch-fidelity)
    "log_every": 50,
    "checkpoint_every": 0,
    "wandb": False,
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base` (returns a new dict)."""
    merged = dict(base)
    for key, value in (override or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compose the run configuration: defaults <- YAML <- CLI overrides."""
    cfg = dict(DEFAULT_CONFIG)
    if path:
        cfg = deep_update(cfg, _read_yaml(path))
    cfg = deep_update(cfg, overrides or {})
    cfg = _apply_task_defaults(cfg)
    return cfg


def _apply_task_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in task-dependent defaults the paper leaves implicit."""
    cfg["task"] = str(cfg.get("task", "inpainting")).lower()
    if cfg["task"] in ("superresolution", "sr", "super-res", "super_res"):
        cfg["task"] = "superres"
    if cfg["task"] not in ("inpainting", "superres"):
        raise ValueError(f"unknown task {cfg['task']!r} (expected 'inpainting' or 'superres')")

    if cfg["task"] == "inpainting":
        # Section 4.1 uses alpha_t = t, beta_t = 1 - t, gamma_t = 0 so that
        # I_0 = x_0 (corrupted) and I_1 = x_1 (clean).
        if not cfg.get("coefficients"):
            cfg["coefficients"] = "inpainting"
        if cfg.get("sigma") is None:
            cfg["sigma"] = 1.0  # noise fills the missing pixels (unit Gaussian)
    else:
        # Section 4.2: m(x1) = U(D(x1)) with alpha_t = 1 - t, beta_t = t, gamma_t = 0.
        if not cfg.get("coefficients"):
            cfg["coefficients"] = "gamma0"
        if cfg.get("sigma") is None:
            cfg["sigma"] = 0.05  # small positive scalar: smooth the base off the manifold
        if cfg.get("low_resolution") is None:
            cfg["low_resolution"] = 64 if int(cfg["resolution"]) >= 256 else int(cfg["resolution"]) // 4
    return cfg


# ---------------------------------------------------------------------------
# Component construction
# ---------------------------------------------------------------------------
def get_coupling(cfg: Dict[str, Any], synthetic: Optional[bool] = None):
    """Build the task coupling from the configuration."""
    from si.couplings import InpaintingCoupling, SuperresCoupling

    if cfg["task"] == "inpainting":
        return InpaintingCoupling(
            sigma=float(cfg["sigma"]),
            num_tiles=int(cfg["num_tiles"]),
            missing_prob=float(cfg["missing_prob"]),
            coefficients=str(cfg["coefficients"]),
        )
    return SuperresCoupling(
        low_res=int(cfg["low_resolution"]),
        sigma=float(cfg["sigma"]),
        down_mode=str(cfg["down_mode"]),
        up_mode=str(cfg["up_mode"]),
        coefficients=str(cfg["coefficients"]),
    )


def build_model(cfg: Dict[str, Any], coupling) -> torch.nn.Module:
    """Instantiate the Appendix-B U-Net velocity model."""
    from si.models import unet_from_config

    in_channels = 3
    conditioning_channels = in_channels if coupling.requires_conditioning else 0
    kwargs: Dict[str, Any] = dict(
        in_channels=in_channels,
        channels=int(cfg["channels"]),
        dim_mults=tuple(cfg["dim_mults"]),
        resnet_block_groups=int(cfg["resnet_block_groups"]),
        num_classes=int(cfg["num_classes"]) if cfg.get("num_classes") else None,
        class_dropout_prob=float(cfg.get("class_dropout_prob", 0.0)),
        conditioning_channels=conditioning_channels,
        learned_sinusoidal_cond=bool(cfg["learned_sinusoidal_cond"]),
        learned_sinusoidal_dim=int(cfg["learned_sinusoidal_dim"]),
        attention_dim_head=int(cfg["attention_dim_head"]),
        attention_heads=int(cfg["attention_heads"]),
        random_fourier_features=bool(cfg["random_fourier_features"]),
        image_size=int(cfg["resolution"]),
        # In-painting: xi * I_t = xi * x1 for all t, so the velocity is
        # structurally zero on observed pixels (Eq. 21 / Section 4.1).
        mask_observed=(cfg["task"] == "inpainting"),
    )
    return unet_from_config(**kwargs)


def build_loss(cfg: Dict[str, Any], coupling):
    """Build the simulation-free velocity-regression objective (Eq. 22)."""
    from si.losses import VelocityLoss

    mask_fn = None
    if cfg["task"] == "inpainting":
        mask_fn = coupling.mask_velocity
    return VelocityLoss(
        coefficients=str(cfg["coefficients"]),
        mask_fn=mask_fn,
        return_dict=True,
    )


def build_interpolant(cfg: Dict[str, Any]):
    from si.interpolants import Interpolant, get_coefficients

    return Interpolant(get_coefficients(str(cfg["coefficients"])))


def build_dataloader(cfg: Dict[str, Any], split: str, fabric=None):
    """ImageNet-1k loader (HuggingFace) at 256/512 with class labels."""
    from si.data import build_dataloader as _build_dataloader

    loader = _build_dataloader(
        split=split,
        batch_size=int(cfg["batch_size"]),
        resolution=int(cfg["resolution"]),
        low_resolution=(
            int(cfg["low_resolution"]) if cfg["task"] == "superres" else None
        ),
        num_workers=int(cfg["num_workers"]),
        synthetic=bool(cfg["synthetic"]),
        cache_dir=cfg.get("cache_dir"),
        streaming=bool(cfg.get("streaming", False)),
        synthetic_length=max(64, int(cfg["batch_size"]) * 8),
        return_dict=True,
    )
    if fabric is not None:
        try:
            loader = fabric.setup_dataloaders(loader)
        except Exception:  # pragma: no cover - single-process fallback
            pass
    return loader


# ---------------------------------------------------------------------------
# Algorithm 1: one gradient step
# ---------------------------------------------------------------------------
def unpack_batch(batch) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return (x1, labels, low_res) from a dataloader batch."""
    if isinstance(batch, dict):
        x1 = batch.get("x1", batch.get("image", batch.get("images")))
        labels = batch.get("label", batch.get("y", batch.get("labels")))
        low = batch.get("low", batch.get("low_res"))
        return x1, labels, low
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            return batch[0], batch[2], batch[1]
        if len(batch) == 2:
            return batch[0], batch[1], None
        return batch[0], None, None
    return batch, None, None


def draw_base_and_interpolant(
    coupling,
    interpolant,
    x1: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Algorithm 1 inner step.

        x0^i = m(x1^i) + sigma zeta^i,   zeta ~ N(0, Id)

    Returns ``(x0, xi, zeta)`` where ``xi`` is the image-shaped conditioning
    (missingness mask / upsampled low-resolution image).  ``zeta`` is returned
    for diagnostics only (``gamma_t = 0`` makes it unnecessary in I_t).
    """
    zeta = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    out = coupling.build_x0(x1, zeta=zeta, generator=generator, return_xi=True)
    if isinstance(out, (tuple, list)):
        x0, xi = out[0], (out[1] if len(out) > 1 else None)
    else:  # pragma: no cover - coupling contract is a tuple
        x0, xi = out, None
    return x0, xi, zeta


@dataclass
class StepStats:
    """Diagnostics accumulated by the training loop (also validates Prop. 3.1)."""

    loss: float = 0.0
    transport_cost: float = 0.0
    grad_norm: float = 0.0
    lr: float = 0.0
    batch_size: int = 0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def train_step(
    model,
    optimizer,
    loss_fn,
    coupling,
    interpolant,
    batch,
    cfg: Dict[str, Any],
    fabric=None,
    generator: Optional[torch.Generator] = None,
    with_transport_cost: bool = True,
) -> StepStats:
    """One iteration of Algorithm 1 (forward loss -> backward -> optimizer)."""
    from si.losses import transport_cost_estimate

    x1, labels, low = unpack_batch(batch)
    device = getattr(fabric, "device", x1.device) if fabric is not None else x1.device
    x1 = x1.to(device, non_blocking=True)
    if labels is not None:
        labels = labels.to(device)

    # x1 ~ rho1(x1); zeta ~ N(0, Id); (masked regions use an independent zeta per channel)
    x0, xi, _ = draw_base_and_interpolant(coupling, interpolant, x1, generator=generator)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Eq. (22): L_hat_b = mean |b_hat_t(I_t)|^2 - 2 I_dot_t . b_hat_t(I_t)
    out = loss_fn(
        model=model,
        x0=x0,
        x1=x1,
        xi=xi,
        y=labels,
        generator=generator,
        return_dict=True,
    )
    loss = out["loss"] if isinstance(out, dict) else out

    if fabric is not None:
        fabric.backward(loss)
    else:
        loss.backward()

    # Appendix B: clip gradient norms at 10,000 (whole parameter vector, PyTorch default 2-norm)
    grad_norm = _clip_gradients(model, float(cfg["grad_clip"]), fabric=fabric)

    optimizer.step()

    stats = StepStats(
        loss=float(loss.detach()),
        grad_norm=float(grad_norm),
        lr=float(optimizer.param_groups[0]["lr"]),
        batch_size=int(x1.shape[0]),
    )
    if with_transport_cost:
        # E[|I_dot_t|^2] diagnostic (Eq. 21): coupled value must beat the independent base.
        with torch.no_grad():
            tc = transport_cost_estimate(x0, x1, interpolant=interpolant)
        stats.transport_cost = float(tc)
    return stats


def _clip_gradients(model, max_norm: float, fabric=None) -> torch.Tensor:
    """Clip the norm of the entire parameter vector (PyTorch's default 2-norm)."""
    if fabric is not None:
        clip = getattr(fabric, "clip_gradients", None)
        if callable(clip):
            try:
                return torch.as_tensor(clip(model, max_norm=max_norm))
            except Exception:  # pragma: no cover - fall through to torch
                pass
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    return torch.nn.utils.clip_grad_norm_(params, max_norm)


def build_optimizer(model, cfg: Dict[str, Any]):
    """Adam, lr 2e-4, no weight decay (Appendix B)."""
    betas = tuple(cfg.get("betas", (0.9, 0.999)))
    return torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        betas=(float(betas[0]), float(betas[1])),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )


def build_scheduler(optimizer, cfg: Dict[str, Any]):
    """StepLR scaling the learning rate by gamma=.99 every 1000 steps (Appendix B)."""
    kind = str(cfg.get("scheduler", "step_lr")).lower()
    if kind in ("step_lr", "steplr", "step"):
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg["scheduler_step_size"]),
            gamma=float(cfg["scheduler_gamma"]),
        )
    if kind in ("none", "constant", ""):
        return None
    raise ValueError(f"unknown scheduler {kind!r}")


# ---------------------------------------------------------------------------
# Sampling / evaluation during training (Algorithm 2)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_batch(model, coupling, cfg: Dict[str, Any], batch, fabric=None) -> Dict[str, torch.Tensor]:
    """Generate X_{t=1} with the Dopri probability-flow ODE (Appendix B)."""
    from si.samplers import ODESampler

    x1, labels, low = unpack_batch(batch)
    device = getattr(fabric, "device", x1.device) if fabric is not None else x1.device
    x1 = x1.to(device)
    if labels is not None:
        labels = labels.to(device)

    sampler = ODESampler(method=str(cfg.get("sample_method", "dopri5")), steps=int(cfg["sample_steps"]))
    x0, xi = coupling.build_x0(x1, return_xi=True)
    sample = sampler(
        model=model,
        x0=x0,
        xi=xi,
        y=labels,
        x1=x1,
        project_observed=(cfg["task"] == "inpainting"),
    )
    out = {"sample": sample, "x0": x0, "x1": x1}
    if xi is not None:
        out["xi"] = xi
    if labels is not None:
        out["label"] = labels
    return out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Algorithm 1 training driver (PyTorch + Lightning Fabric)."""

    def __init__(self, cfg: Dict[str, Any], fabric=None):
        self.cfg = cfg
        self.fabric = fabric
        self.device = getattr(fabric, "device", None)
        self.is_main = dist_utils.is_main_process(fabric)

        seed = int(cfg.get("seed", 0))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

        self.coupling = get_coupling(cfg)
        self.interpolant = build_interpolant(cfg)
        self.model = build_model(cfg, self.coupling)
        self.loss_fn = build_loss(cfg, self.coupling)

        if fabric is not None:
            try:
                self.model = fabric.setup_module(self.model)
            except Exception:  # pragma: no cover
                pass
            self.model = self.model.to(getattr(fabric, "device", "cpu"))

        self.base_model = getattr(self.model, "module", self.model)
        self.optimizer = build_optimizer(self.model, cfg)
        if fabric is not None:
            try:
                self.optimizer = fabric.setup_optimizers(self.optimizer)
            except Exception:  # pragma: no cover
                pass
        self.scheduler = build_scheduler(self.optimizer, cfg)

        # EMA is not mentioned in the paper; kept optional for FID stability.
        self.ema = None
        if cfg.get("use_ema"):
            try:
                from si.utils.ema import EMA

                self.ema = EMA(self.base_model, decay=float(cfg.get("ema_decay", 0.9999)))
            except Exception:  # pragma: no cover
                self.ema = None

        os.makedirs(self.run_dir, exist_ok=True)

    # -- paths ---------------------------------------------------------
    @property
    def run_dir(self) -> str:
        name = self.cfg.get("run_name") or f"{self.cfg['task']}_{self.cfg['resolution']}"
        return os.path.join(str(self.cfg.get("out_dir", "runs")), str(name))

    # -- checkpointing -------------------------------------------------
    def save(self, step: int, tag: Optional[str] = None) -> str:
        path = os.path.join(self.run_dir, tag or f"checkpoint_{step:07d}.pt")
        payload = {
            "step": step,
            "config": self.cfg,
            "model": self.base_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.ema is not None:
            payload["ema"] = self.ema.state_dict()
        torch.save(payload, path)
        return path

    @staticmethod
    def load_into(model, path: str, map_location=None, use_ema: bool = True) -> Dict[str, Any]:
        payload = torch.load(path, map_location=map_location or "cpu", weights_only=False)
        state = payload.get("ema") if (use_ema and "ema" in payload) else payload.get("model", payload)
        target = getattr(model, "module", model)
        missing, unexpected = target.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning("checkpoint load: missing=%d unexpected=%d", len(missing), len(unexpected))
        return payload

    # -- main loop ------------------------------------------------------
    def fit(self) -> Dict[str, Any]:
        cfg = self.cfg
        total_steps = int(cfg["steps"])
        accumulation = max(1, int(cfg.get("grad_accum_steps", 1)))
        loader = build_dataloader(cfg, str(cfg.get("train_split", "train")), fabric=self.fabric)
        valid_loader = None
        if int(cfg.get("eval_every", 0)) > 0 or int(cfg.get("sample_every", 0)) > 0:
            valid_loader = build_dataloader(cfg, str(cfg.get("valid_split", "validation")), fabric=self.fabric)

        self.model.train()
        step = 0
        history: List[Dict[str, float]] = []
        epoch = 0
        started = time.time()

        while step < total_steps:
            epoch += 1
            for batch in loader:
                stats = train_step(
                    model=self.model,
                    optimizer=self.optimizer,
                    loss_fn=self.loss_fn,
                    coupling=self.coupling,
                    interpolant=self.interpolant,
                    batch=batch,
                    cfg=cfg,
                    fabric=self.fabric,
                    generator=self.generator,
                )
                if self.ema is not None:
                    self.ema.update(self.base_model, step=step)
                if self.scheduler is not None:
                    self.scheduler.step()
                step += 1

                if self.is_main and int(cfg.get("log_every", 50)) > 0 and step % int(cfg["log_every"]) == 0:
                    record = stats.as_dict()
                    record["step"] = step
                    record["elapsed"] = time.time() - started
                    history.append(record)
                    logger.info(
                        "step %6d/%d | loss %.4f | E|I_dot|^2 %.4f | grad_norm %.3f | lr %.3e",
                        step, total_steps, stats.loss, stats.transport_cost,
                        stats.grad_norm, stats.lr,
                    )

                if valid_loader is not None and int(cfg.get("sample_every", 0)) > 0 \
                        and step % int(cfg["sample_every"]) == 0:
                    self.qualitative(valid_loader, step)
                    self.model.train()

                if valid_loader is not None and int(cfg.get("eval_every", 0)) > 0 \
                        and step % int(cfg["eval_every"]) == 0:
                    self.evaluate(valid_loader, step)

                if int(cfg.get("checkpoint_every", 0)) > 0 and step % int(cfg["checkpoint_every"]) == 0:
                    if self.is_main:
                        logger.info("saved %s", self.save(step))

                if step >= total_steps:
                    break

        final_path = None
        if self.is_main:
            final_path = self.save(step, tag="checkpoint_final.pt")
            logger.info("training finished after %d steps; checkpoint: %s", step, final_path)

        return {
            "steps": step,
            "checkpoint": final_path,
            "history": history,
            "run_dir": self.run_dir,
        }

    # -- periodic evaluation -------------------------------------------
    def _eval_model(self):
        """Model used for sampling: EMA weights when available."""
        if self.ema is None:
            return self.model
        try:
            self.ema.store(self.base_model)
            self.ema.copy_to(self.base_model)
            return self.model
        except Exception:  # pragma: no cover
            return self.model

    def _restore_model(self) -> None:
        if self.ema is None:
            return
        try:
            self.ema.restore(self.base_model)
        except Exception:  # pragma: no cover
            pass

    @torch.no_grad()
    def qualitative(self, loader, step: int) -> Optional[str]:
        """Save base/model/ground-truth triples (Figs. 3, 4, 6)."""
        model = self._eval_model()
        model.eval()
        batch = next(iter(loader))
        out = sample_batch(model, self.coupling, self.cfg, batch, fabric=self.fabric)
        path = os.path.join(self.run_dir, f"qualitative_{step:07d}.pt")
        if self.is_main:
            torch.save({k: v.cpu() for k, v in out.items()}, path)
            logger.info("qualitative samples saved to %s", path)
        self._restore_model()
        return path

    @torch.no_grad()
    def evaluate(self, loader, step: int) -> Optional[Dict[str, float]]:
        """FID-50k evaluation hook (Table 2 / Table 3)."""
        try:
            from eval.fid import evaluate_fid_50k
        except Exception as exc:  # pragma: no cover
            logger.warning("FID evaluation unavailable (%s)", exc)
            return None
        model = self._eval_model()
        model.eval()

        num = int(self.cfg.get("fid_num_samples", 50000))
        per_batch = int(self.cfg["batch_size"])
        generated: List[torch.Tensor] = []
        produced = 0
        while produced < num:
            batch = next(iter(loader))
            out = sample_batch(model, self.coupling, self.cfg, batch, fabric=self.fabric)
            generated.append(out["sample"].detach().cpu())
            produced += int(out["sample"].shape[0])
        samples = torch.cat(generated, dim=0)[:num]
        task = "superres_64_256" if self.cfg["task"] == "superres" else "inpainting"
        result = evaluate_fid_50k(samples, task=task, batch_size=per_batch)
        self._restore_model()
        payload = result.as_dict() if hasattr(result, "as_dict") else {"fid": float(result)}
        if self.is_main:
            logger.info("step %d FID: %s", step, payload)
        return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Algorithm 1 training for stochastic interpolants with data-dependent couplings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config (configs/*.yaml)")
    parser.add_argument("--task", type=str, default=None, choices=["inpainting", "superres"])
    parser.add_argument("--resolution", type=int, default=None, help="256 or 512")
    parser.add_argument("--low-resolution", type=int, default=None, help="64 or 256 (SR conditioning)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help="gradient steps (paper: 200000)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--coefficients", type=str, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--num-tiles", type=int, default=None)
    parser.add_argument("--missing-prob", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / cuda:0")
    parser.add_argument("--gpus", type=int, default=1, help="number of processes for Fabric")
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true", help="random tensors instead of ImageNet")
    parser.add_argument("--no-ema", action="store_true", help="disable the (optional) EMA shadow weights")
    parser.add_argument("--dry-run", action="store_true", help="build everything, run 1 step, exit")
    return parser.parse_args(argv)


_CLI_TO_CONFIG = {
    "task": "task",
    "resolution": "resolution",
    "low_resolution": "low_resolution",
    "batch_size": "batch_size",
    "steps": "steps",
    "lr": "lr",
    "sigma": "sigma",
    "coefficients": "coefficients",
    "channels": "channels",
    "grad_clip": "grad_clip",
    "num_tiles": "num_tiles",
    "missing_prob": "missing_prob",
    "seed": "seed",
    "out_dir": "out_dir",
    "run_name": "run_name",
    "num_workers": "num_workers",
    "log_every": "log_every",
    "sample_every": "sample_every",
    "eval_every": "eval_every",
    "checkpoint_every": "checkpoint_every",
    "precision": "precision",
}


def config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for attr, key in _CLI_TO_CONFIG.items():
        value = getattr(args, attr, None)
        if value is not None:
            overrides[key] = value
    if args.synthetic:
        overrides["synthetic"] = True
    if args.no_ema:
        overrides["use_ema"] = False
    return build_config(args.config, overrides)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = config_from_args(args)
    if args.dry_run:
        cfg["steps"] = 1
        cfg["log_every"] = 1

    # PyTorch + Lightning Fabric for parallelism (Appendix B, "Miscellaneous").
    fabric = _setup_fabric(args, cfg)
    trainer = Trainer(cfg, fabric=fabric)

    if trainer.is_main:
        logger.info("task=%s resolution=%s low_resolution=%s",
                    cfg["task"], cfg["resolution"], cfg["low_resolution"])
        logger.info("coupling=%s coefficients=%s sigma=%s",
                    type(trainer.coupling).__name__, cfg["coefficients"], cfg["sigma"])
        logger.info("model parameters: %d", sum(p.numel() for p in trainer.base_model.parameters()))
        logger.info("Algorithm 1: batch_size=%d steps=%d lr=%g clip=%.0f (gamma=%.2f every %d)",
                    cfg["batch_size"], cfg["steps"], cfg["lr"], cfg["grad_clip"],
                    cfg["scheduler_gamma"], cfg["scheduler_step_size"])

    if args.resume:
        Trainer.load_into(trainer.base_model, args.resume, map_location="cpu")
        if trainer.is_main:
            logger.info("resumed weights from %s", args.resume)

    result = trainer.fit()

    if trainer.is_main:
        _write_yaml(cfg, os.path.join(trainer.run_dir, "config.yaml"))
        logger.info("done: %s", json.dumps({k: v for k, v in result.items() if k != "history"}, default=str))
    return 0


def _setup_fabric(args: argparse.Namespace, cfg: Dict[str, Any]):
    """Create the Fabric (or single-process shim) for training."""
    device = args.device
    if device is None and not torch.cuda.is_available():
        device = "cpu"
    try:
        fabric = dist_utils.setup(
            accelerator="auto" if device is None else ("cpu" if device == "cpu" else "gpu"),
            devices=int(args.gpus) if args.gpus else "auto",
            precision=str(args.precision or cfg.get("precision", "32-true")),
            seed=cfg.get("seed"),
        )
        if device == "cpu" and hasattr(fabric, "to_device"):
            pass
        return fabric
    except Exception as exc:  # pragma: no cover - fallback keeps CPU runs alive
        logger.warning("Fabric setup failed (%s); continuing single-process", exc)
        return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
