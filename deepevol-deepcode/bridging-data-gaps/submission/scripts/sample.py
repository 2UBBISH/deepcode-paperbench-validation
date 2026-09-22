#!/usr/bin/env python
"""Sample / generate images with an adapted DPMs-ANT model.

This command-line entry point loads a *frozen* pretrained diffusion backbone
(DDPM 256x256 pixel space or LDM 64x64 latent space) with zero-initialized
adaptors inserted into the U-Net shift modules, optionally restores a trained
adaptor checkpoint, and then runs the adapted reverse process of Eq. (2)/(3)
(DDIM ``eta=0`` or stochastic DDPM ``eta=1``) to generate images.

For LDM backbones the sampling happens in the 64x64 latent space and the frozen
autoencoder decodes the latents back to 256x256 pixels (Rombach et al. 2022),
as described in Section 5.2 Configurations.

Examples
--------
::

    # 1000 DDIM samples with a trained adaptor checkpoint
    python scripts/sample.py --task ffhq_sunglasses --backbone ddpm \\
        --adaptor-checkpoint runs/ddpm_ant_ffhq_sunglasses/adaptor.pt \\
        --num-samples 1000 --num-steps 100 --out samples/ffhq_sunglasses

    # LDM latent sampling, decoded with the frozen autoencoder
    python scripts/sample.py --task ffhq_babies_ldm --backbone ldm \\
        --ldm-checkpoint models/ldm/ldm_256.ckpt \\
        --autoencoder-checkpoint models/ldm/autoencoder.ckpt \\
        --adaptor-checkpoint runs/ldm_ant_ffhq_babies/adaptor.pt \\
        --num-samples 1000 --num-steps 100 --out samples/ffhq_babies_ldm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

_LOGGER_NAME = "dpm_ant.sample"
LOGGER = logging.getLogger(_LOGGER_NAME)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_CONFIG = os.path.join(_ROOT, "configs", "default.yaml")
PER_TASK_CONFIG = os.path.join(_ROOT, "configs", "per_task.yaml")


# ---------------------------------------------------------------------------
# config helpers (mirrors scripts/train_ant.py for consistent precedence)
# ---------------------------------------------------------------------------
def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        import yaml

        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Could not read config %s: %s", path, exc)
        return {}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    config_path: Optional[str] = None,
    per_task_path: Optional[str] = None,
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge ``default.yaml`` + ``per_task.yaml``; flatten per-task defaults."""
    cfg = _load_yaml(config_path or DEFAULT_CONFIG)
    per_task = _load_yaml(per_task_path or PER_TASK_CONFIG)
    cfg = _deep_update(cfg, per_task)

    # `per_task.yaml` keeps a `defaults` block that mirrors the global defaults.
    defaults = cfg.get("defaults")
    if isinstance(defaults, dict):
        cfg.setdefault("ant", {})
        cfg["ant"] = _deep_update(defaults.copy(), cfg.get("ant", {}))
        if isinstance(defaults.get("adaptor"), dict):
            cfg["adaptor"] = _deep_update(defaults["adaptor"], cfg.get("adaptor", {}))

    if task:
        cfg.setdefault("task", task)
    return cfg


def _task_entry(cfg: Dict[str, Any], task: Optional[str]) -> Dict[str, Any]:
    if not task:
        return {}
    tasks = cfg.get("tasks") or {}
    entry = tasks.get(task) or {}
    return entry if isinstance(entry, dict) else {}


def _ensure_sampling_block(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fill the ``sampling`` block with paper defaults (Section 5.2)."""
    sampling = cfg.setdefault("sampling", {})
    defaults = {
        "method": "ddim",
        "num_steps": 100,
        "eta": 0.0,
        "num_timesteps": 1000,
        "schedule": "linear",
        "batch_size": 16,
        "num_samples": 1000,
        "use_classifier_guidance": False,
        "guidance_scale": 5.0,
        "classifier_target_index": 1,
        "clip_denoised": True,
        "decode_latents": True,
        "save_images": True,
    }
    for key, value in defaults.items():
        sampling.setdefault(key, value)
    return sampling


def _resolve_backbone(cfg: Dict[str, Any], cli_value: Optional[str], task: Optional[str]) -> str:
    if cli_value:
        return cli_value
    entry = _task_entry(cfg, task)
    if entry.get("backbone"):
        return entry["backbone"]
    models = cfg.get("models") or {}
    if any(k.startswith("ldm") for k in (cfg.get("tasks") or {})):
        pass
    if "ldm" in models and task and str(task).endswith("_ldm"):
        return "ldm"
    return "ddpm"


def _checkpoint_path(*candidates: Optional[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    for candidate in candidates:
        if candidate:
            return candidate
    return None


# ---------------------------------------------------------------------------
# model / classifier / data construction
# ---------------------------------------------------------------------------
def build_schedule_from_cfg(cfg: Dict[str, Any], backbone: str = "ddpm"):
    from dpm_ant.diffusion.schedule import build_schedule

    diffusion_cfg = dict(cfg.get("diffusion") or {})
    sampling_cfg = dict(cfg.get("sampling") or {})
    for key in ("num_timesteps", "schedule", "beta_start", "beta_end", "cosine_s"):
        if key in sampling_cfg and key not in diffusion_cfg:
            diffusion_cfg[key] = sampling_cfg[key]
    return build_schedule(diffusion_cfg)


def build_backbone(
    cfg: Dict[str, Any],
    backbone: str,
    task: Optional[str],
    device: str = "cpu",
    checkpoint: Optional[str] = None,
    autoencoder_checkpoint: Optional[str] = None,
    insert: bool = True,
    verbose: bool = True,
) -> Tuple[Any, Optional[Any]]:
    """Load the frozen (adapted) backbone; returns ``(model, autoencoder)``."""
    from dpm_ant.models.adaptor import build_adaptor_factory

    models_cfg = cfg.get("models") or {}
    adaptor_cfg = dict(cfg.get("adaptor") or {})
    adaptor_cfg["backbone"] = backbone

    if backbone == "ldm":
        from dpm_ant.models.ldm_loader import (
            FrozenLDMAutoencoder,
            FrozenLDMUNet,
            insert_adaptors_ldm,
            load_autoencoder,
            load_ldm_unet,
        )

        ldm_cfg = dict(models_cfg.get("ldm") or {})
        ckpt = _checkpoint_path(
            checkpoint,
            ldm_cfg.get("ckpt"),
            ldm_cfg.get("checkpoint"),
            ldm_cfg.get("ldm_ckpt"),
        )
        unet = load_ldm_unet(checkpoint=ckpt, cfg=ldm_cfg, device=device, verbose=verbose)
        factory = build_adaptor_factory(adaptor_cfg, backbone="ldm", task=task)
        if insert:
            try:
                insert_adaptors_ldm(
                    unet,
                    adaptor_factory=factory,
                    shift_only=bool(adaptor_cfg.get("insert_into", "shift") == "shift"),
                    verbose=verbose,
                )
            except TypeError:  # tolerate alternate signatures
                insert_adaptors_ldm(unet, adaptor_factory=factory)
        model = unet if isinstance(unet, FrozenLDMUNet) else FrozenLDMUNet(unet)

        autoencoder = None
        ae_cfg = dict(models_cfg.get("autoencoder") or ldm_cfg.get("autoencoder") or {})
        ae_ckpt = _checkpoint_path(
            autoencoder_checkpoint,
            ldm_cfg.get("autoencoder_ckpt"),
            ae_cfg.get("ckpt"),
            ae_cfg.get("checkpoint"),
        )
        try:
            ae = load_autoencoder(checkpoint=ae_ckpt, cfg=ae_cfg, device=device, verbose=verbose)
            autoencoder = ae if isinstance(ae, FrozenLDMAutoencoder) else FrozenLDMAutoencoder(ae)
        except Exception as exc:  # pragma: no cover - optional component
            LOGGER.warning("Could not load LDM autoencoder: %s", exc)
        if hasattr(model, "set_autoencoder"):
            try:
                model.set_autoencoder(autoencoder)
            except Exception:  # pragma: no cover
                pass
        return model, autoencoder

    # ---------------- DDPM (guided-diffusion 256x256) ----------------
    from dpm_ant.models.unet_loader import (
        FrozenDDPMUNet,
        insert_adaptors,
        load_guided_diffusion_unet,
    )

    ddpm_cfg = dict(models_cfg.get("ddpm") or {})
    ckpt = _checkpoint_path(
        checkpoint,
        ddpm_cfg.get("ckpt"),
        ddpm_cfg.get("checkpoint"),
        ddpm_cfg.get("diffusion_ckpt"),
    )
    unet = load_guided_diffusion_unet(checkpoint=ckpt, cfg=ddpm_cfg, device=device, verbose=verbose)
    factory = build_adaptor_factory(adaptor_cfg, backbone="ddpm", task=task)
    if insert:
        try:
            insert_adaptors(
                unet,
                adaptor_factory=factory,
                shift_only=bool(adaptor_cfg.get("insert_into", "shift") == "shift"),
                verbose=verbose,
            )
        except TypeError:
            insert_adaptors(unet, adaptor_factory=factory)
    model = unet if isinstance(unet, FrozenDDPMUNet) else FrozenDDPMUNet(unet)
    return model, None


def build_classifier_for_task(
    cfg: Dict[str, Any],
    backbone: str,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    verbose: bool = True,
):
    """Load the frozen 2-way source/target classifier (or ``None``)."""
    from dpm_ant.models.classifier import build_classifier

    clf_cfg = dict(cfg.get("classifier") or {})
    models_cfg = cfg.get("models") or {}
    bb_cfg = dict(models_cfg.get(("ldm" if backbone == "ldm" else "ddpm")) or {})
    ckpt = _checkpoint_path(
        checkpoint,
        clf_cfg.get("ckpt"),
        clf_cfg.get("checkpoint"),
        bb_cfg.get("classifier_ckpt"),
        bb_cfg.get("classifier_checkpoint"),
    )
    if not ckpt:
        LOGGER.warning("No classifier checkpoint found; sampling without guidance.")
        return None
    try:
        return build_classifier(
            cfg=cfg,
            backbone=backbone,
            checkpoint=ckpt,
            device=device,
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to build classifier (%s); sampling without guidance.", exc)
        return None


def load_adaptor_checkpoint(model: Any, path: Optional[str], device: str = "cpu") -> Dict[str, Any]:
    """Restore a trained adaptor checkpoint onto the model, if provided."""
    info: Dict[str, Any] = {"loaded": False, "path": path}
    if not path:
        return info
    if not os.path.isfile(path):
        LOGGER.warning("Adaptor checkpoint %s not found; sampling from the pretrained model.", path)
        return info

    import torch

    payload = torch.load(path, map_location=device)
    state = payload
    if isinstance(payload, dict):
        for key in ("adaptor", "adaptor_state_dict", "state_dict", "model"):
            inner = payload.get(key)
            if isinstance(inner, dict):
                state = inner
                break
    if not isinstance(state, dict):  # pragma: no cover - defensive
        LOGGER.warning("Unrecognised adaptor checkpoint format in %s.", path)
        return info

    target = model
    if hasattr(model, "load_adaptor_state_dict"):
        try:
            model.load_adaptor_state_dict(state, strict=False)
            info.update({"loaded": True, "num_tensors": len(state)})
            if isinstance(payload, dict):
                info["iteration"] = payload.get("iteration", payload.get("step"))
            return info
        except Exception as exc:  # pragma: no cover - fall through to raw load
            LOGGER.warning("load_adaptor_state_dict failed (%s); trying raw load.", exc)

    module = getattr(target, "unet", None) or getattr(target, "model", None) or target
    if hasattr(module, "load_state_dict"):
        result = module.load_state_dict(state, strict=False)
        info.update(
            {
                "loaded": True,
                "num_tensors": len(state),
                "missing_keys": list(getattr(result, "missing_keys", []) or []),
                "unexpected_keys": list(getattr(result, "unexpected_keys", []) or []),
            }
        )
        if isinstance(payload, dict):
            info["iteration"] = payload.get("iteration", payload.get("step"))
    return info


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def sample_to_dir(
    model: Any,
    classifier: Optional[Any],
    autoencoder: Optional[Any],
    cfg: Dict[str, Any],
    backbone: str,
    task: Optional[str],
    device: str = "cpu",
    out_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    num_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    eta: Optional[float] = None,
    method: Optional[str] = None,
    seed: Optional[int] = None,
    use_guidance: Optional[bool] = None,
    guidance_scale: Optional[float] = None,
    latent_size: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Generate images (or latents) and write them to ``out_dir``."""
    import torch

    from dpm_ant.sampling.sampler import build_sampler

    sampling_cfg = dict(cfg.get("sampling") or {})
    if method is not None:
        sampling_cfg["method"] = method
    if num_steps is not None:
        sampling_cfg["num_steps"] = num_steps
    if batch_size is not None:
        sampling_cfg["batch_size"] = batch_size
    if eta is not None:
        sampling_cfg["eta"] = eta
    if num_samples is not None:
        sampling_cfg["num_samples"] = num_samples
    if seed is not None:
        sampling_cfg["seed"] = seed
    if use_guidance is not None:
        sampling_cfg["use_classifier_guidance"] = use_guidance
    if guidance_scale is not None:
        sampling_cfg["guidance_scale"] = guidance_scale

    if backbone == "ldm":
        models_cfg = cfg.get("models") or {}
        ldm_cfg = dict(models_cfg.get("ldm") or {})
        sampling_cfg.setdefault("latent_size", latent_size or ldm_cfg.get("latent_size", 64))
        sampling_cfg.setdefault("latent_channels", ldm_cfg.get("latent_channels", 4))
    else:
        models_cfg = cfg.get("models") or {}
        ddpm_cfg = dict(models_cfg.get("ddpm") or {})
        image_size = ddpm_cfg.get("image_size", (cfg.get("data") or {}).get("image_size", 256))
        sampling_cfg.setdefault("image_size", image_size)

    sampler = build_sampler(
        cfg=sampling_cfg,
        model=model,
        classifier=classifier,
        autoencoder=autoencoder,
        backbone=backbone,
        task=task,
        device=device,
    )

    t_start = time.time()
    result: Dict[str, Any] = {
        "backbone": backbone,
        "task": task,
        "num_samples": sampling_cfg.get("num_samples", 1000),
        "num_steps": sampling_cfg.get("num_steps", 100),
        "method": sampling_cfg.get("method", "ddim"),
        "eta": sampling_cfg.get("eta"),
        "out_dir": out_dir,
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        try:
            sampler.sample_to_dir(out_dir=out_dir, verbose=verbose)
            result["saved"] = True
        except Exception as exc:
            LOGGER.warning("sample_to_dir failed (%s); falling back to tensor sampling.", exc)
            images = sampler.sample(verbose=verbose)
            result["saved"] = _save_tensor(images, out_dir, backbone)
    else:
        images = sampler.sample(verbose=verbose)
        result["shape"] = list(images.shape)

    result["elapsed_sec"] = time.time() - t_start
    result["elapsed_hours"] = result["elapsed_sec"] / 3600.0
    return result


def _save_tensor(images, out_dir: str, backbone: str) -> bool:
    """Save a generated tensor as PNG files (latents are decoded first)."""
    import torch

    out_dir = os.path.join(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    try:
        from dpm_ant.data.datasets import denormalize_images
    except Exception:  # pragma: no cover - defensive
        def denormalize_images(x):
            return (x.clamp(-1, 1) + 1.0) / 2.0

    images = images.detach().cpu().float()
    if images.dim() == 4 and images.min() < -0.01:
        images = denormalize_images(images)
    images = images.clamp(0.0, 1.0)

    try:
        from PIL import Image
        import numpy as np

        for idx, img in enumerate(images):
            arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
            Image.fromarray(arr).save(os.path.join(out_dir, f"{idx:06d}.png"))
        return True
    except Exception as exc:  # pragma: no cover - fallback
        LOGGER.warning("PNG export unavailable (%s); writing .pt tensor.", exc)
        torch.save(images, os.path.join(out_dir, "samples.pt"))
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample images from an adapted DPMs-ANT model (DDPM/LDM)."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to configs/default.yaml")
    parser.add_argument("--per-task-config", default=PER_TASK_CONFIG, help="Path to configs/per_task.yaml")
    parser.add_argument("--task", default=None, help="Task key, e.g. ffhq_sunglasses")
    parser.add_argument("--backbone", default=None, choices=["ddpm", "ldm"], help="Diffusion backbone")
    parser.add_argument("--method", default=None, choices=["ddim", "ddpm"], help="Reverse process (eta=0/1)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of images (default 1000)")
    parser.add_argument("--num-steps", type=int, default=None, help="Reverse steps (default 100 for DDIM)")
    parser.add_argument("--batch-size", type=int, default=None, help="Sampling batch size")
    parser.add_argument("--eta", type=float, default=None, help="DDIM eta (0 deterministic, 1 DDPM)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:0")
    parser.add_argument("--diffusion-checkpoint", default=None, help="Frozen DDPM U-Net checkpoint")
    parser.add_argument("--ldm-checkpoint", default=None, help="LDM U-Net checkpoint")
    parser.add_argument("--autoencoder-checkpoint", default=None, help="LDM autoencoder checkpoint")
    parser.add_argument("--classifier-checkpoint", default=None, help="Fine-tuned 2-way classifier checkpoint")
    parser.add_argument("--adaptor-checkpoint", default=None, help="Trained adaptor checkpoint (psi)")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Classifier guidance scale")
    parser.add_argument("--classifier-guidance", action="store_true", help="Enable classifier-guided sampling")
    parser.add_argument("--out", default=None, help="Output directory for generated images")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config, args.per_task_config, args.task)
    task_entry = _task_entry(cfg, args.task)
    backbone = _resolve_backbone(cfg, args.backbone, args.task)
    cfg = _deep_update(cfg, {"task_entry": task_entry}) if task_entry else cfg
    _ensure_sampling_block(cfg)

    device = args.device or cfg.get("device", "cpu")
    try:
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA unavailable; falling back to CPU.")
            device = "cpu"
    except Exception:  # pragma: no cover
        device = "cpu"

    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    try:
        import torch

        torch.manual_seed(int(seed))
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(int(seed))
    except Exception:  # pragma: no cover
        pass

    models_cfg = cfg.get("models") or {}
    if backbone == "ldm":
        ldm_cfg = dict(models_cfg.get("ldm") or {})
        diffusion_ckpt = _checkpoint_path(args.ldm_checkpoint, ldm_cfg.get("ckpt"), ldm_cfg.get("checkpoint"))
    else:
        ddpm_cfg = dict(models_cfg.get("ddpm") or {})
        diffusion_ckpt = _checkpoint_path(
            args.diffusion_checkpoint,
            ddpm_cfg.get("ckpt"),
            ddpm_cfg.get("checkpoint"),
            ddpm_cfg.get("diffusion_ckpt"),
        )

    if args.dry_run:
        payload = {
            "task": args.task,
            "backbone": backbone,
            "device": device,
            "seed": seed,
            "diffusion_checkpoint": diffusion_ckpt,
            "autoencoder_checkpoint": args.autoencoder_checkpoint,
            "classifier_checkpoint": args.classifier_checkpoint,
            "adaptor_checkpoint": args.adaptor_checkpoint,
            "sampling": cfg.get("sampling", {}),
            "adaptor": cfg.get("adaptor", {}),
            "out": args.out,
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    # --- build model & restore adaptor weights -------------------------------
    model, autoencoder = build_backbone(
        cfg,
        backbone,
        args.task,
        device=device,
        checkpoint=diffusion_ckpt,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        insert=True,
        verbose=args.verbose,
    )

    adaptor_info = load_adaptor_checkpoint(model, args.adaptor_checkpoint, device=device)

    # --- optional classifier for Eq. (4) guided sampling ---------------------
    classifier = None
    if args.classifier_guidance or args.classifier_checkpoint:
        classifier = build_classifier_for_task(
            cfg,
            backbone,
            checkpoint=args.classifier_checkpoint,
            device=device,
            verbose=args.verbose,
        )

    out_dir = args.out or os.path.join(
        cfg.get("logging", {}).get("sample_dir", os.path.join(_ROOT, "samples")),
        args.task or "default",
    )

    report = sample_to_dir(
        model=model,
        classifier=classifier,
        autoencoder=autoencoder,
        cfg=cfg,
        backbone=backbone,
        task=args.task,
        device=device,
        out_dir=out_dir,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        eta=args.eta,
        method=args.method,
        seed=seed,
        use_guidance=True if args.classifier_guidance else None,
        guidance_scale=args.guidance_scale,
        verbose=args.verbose,
    )
    report["adaptor"] = adaptor_info
    report["seed"] = seed

    meta_path = os.path.join(out_dir, "sampling_meta.json")
    try:
        with open(meta_path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        LOGGER.info("Wrote sampling metadata to %s", meta_path)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Could not write sampling metadata: %s", exc)

    LOGGER.info(
        "Generated %s samples (%s, %s steps) in %.1f s -> %s",
        report.get("num_samples"),
        report.get("method"),
        report.get("num_steps"),
        report.get("elapsed_sec", 0.0),
        out_dir,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
