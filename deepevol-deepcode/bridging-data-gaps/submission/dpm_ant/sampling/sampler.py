"""Adapted reverse sampling for DPMs-ANT.

Implements the reverse (generation) process of the adapted diffusion model
``eps_{theta, psi}`` for both DDPM (eta = 1) and DDIM (eta = 0) sampling:

Eq. (2) of the paper (reverse process):
    x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * eps_theta(x_t, t))
              + sigma_t * z,   z ~ N(0, I),  sigma_t^2 = ...
Eq. (3) of the paper (DDIM / eta-parameterized reverse step):
    x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0_hat
              + sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * eps_theta(x_t, t)
              + sigma_t * z,
    sigma_t = eta * sqrt((1 - alpha_bar_{t-1}) / (1 - alpha_bar_t))
                     * sqrt(1 - alpha_bar_t / alpha_bar_{t-1})
with ``eta = 0`` giving deterministic DDIM sampling and ``eta = 1`` recovering DDPM.

Additionally supports conditional (classifier-guided) sampling, Eq. (4):
    eps_hat = eps_theta(x_t, t) - sigma_hat_t^2 * gamma * grad_{x_t} log p_phi(y=T|x_t)

For LDM backbones, sampling happens in the 64x64 latent space and the final
latent is decoded with the frozen autoencoder (Rombach et al. 2022).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..diffusion.schedule import NoiseSchedule, build_schedule

__all__ = [
    "SamplingConfig",
    "Sampler",
    "DDIMSampler",
    "build_sampler",
    "sample_images",
    "ddim_sample",
    "ddpm_sample",
]

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _extract(arr: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Gather schedule values at per-sample timesteps and broadcast to ``ndim`` dims."""
    t = torch.as_tensor(t, device=arr.device)
    t = t.long().reshape(-1)
    out = arr.to(t.device).gather(0, t)
    while out.dim() < ndim:
        out = out.unsqueeze(-1)
    return out


def _resolve_callable(module: Any) -> Callable:
    """Return the noise-prediction callable of a backbone wrapper or raw module."""
    if module is None:
        raise ValueError("A diffusion model must be provided for sampling.")
    if isinstance(module, nn.Module):
        for name in ("epsilon_theta", "predict_noise", "predict_eps", "denoise"):
            fn = getattr(module, name, None)
            if callable(fn):
                return fn
    if callable(module):
        return module
    raise TypeError(f"Cannot resolve a noise-prediction callable from {type(module)!r}")


def _unwrap_prediction(out: Any) -> torch.Tensor:
    """Handle ``learn_sigma`` tuples / dict outputs, returning the noise branch."""
    if isinstance(out, dict):
        for key in ("eps", "epsilon", "noise", "pred"):
            if key in out:
                return out[key]
        out = next(iter(out.values()))
    if isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, torch.Tensor) and out.dim() == 4 and out.shape[1] % 2 == 0:
        # `learn_sigma=True` models predict concatenated [eps, sigma]; wrappers
        # already split, but be defensive here for raw modules.
        # Detect by comparing against in_channels only when unambiguous.
        pass
    return out


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SamplingConfig:
    """Sampling hyper-parameters (Section 5.2 Configurations)."""

    method: str = "ddim"          # "ddim" (eta=0) or "ddpm" (eta=1)
    num_steps: int = 100          # DDIM steps used for evaluation
    eta: Optional[float] = None   # None -> 0.0 for ddim, 1.0 for ddpm
    num_timesteps: int = 1000     # training-time T (diffusion schedule)
    schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    cosine_s: float = 0.008
    batch_size: int = 16
    num_samples: int = 1000
    use_classifier_guidance: bool = False
    guidance_scale: float = 5.0
    classifier_target_index: int = 1
    clip_denoised: bool = True
    dynamic_threshold: bool = False
    device: str = "cuda"
    dtype: str = "float32"
    seed: Optional[int] = None
    # task metadata
    backbone: str = "ddpm"        # "ddpm" | "ldm"
    unconditional_guidance_scale: float = 1.0

    @staticmethod
    def from_dict(cfg: Optional[Dict[str, Any]] = None, backbone: str = "ddpm",
                  task: Optional[str] = None, **overrides) -> "SamplingConfig":
        cfg = dict(cfg or {})
        flat: Dict[str, Any] = {}

        # nested blocks found in configs/default.yaml
        sampling = cfg.get("sampling", {}) or {}
        diffusion = cfg.get("diffusion", {}) or {}
        classifier = cfg.get("classifier", {}) or {}
        defaults = cfg.get("defaults", {}) or {}

        flat.update({k: v for k, v in sampling.items()})
        if "num_timesteps" in diffusion:
            flat.setdefault("num_timesteps", diffusion["num_timesteps"])
        if "T" in diffusion:
            flat.setdefault("num_timesteps", diffusion["T"])
        for key in ("schedule", "beta_schedule"):
            if key in diffusion:
                flat.setdefault("schedule", diffusion[key])
        for key in ("beta_start", "beta_end", "cosine_s", "eta"):
            if key in diffusion and key != "eta":
                flat.setdefault(key, diffusion[key])
        if "target_index" in classifier:
            flat.setdefault("classifier_target_index", classifier["target_index"])

        for key in ("batch_size", "num_timesteps", "schedule", "eta", "seed", "backbone"):
            if key in defaults:
                flat.setdefault(key, defaults[key])

        # per-task overrides
        tasks = cfg.get("tasks", {}) or {}
        if task is not None and task in tasks:
            tcfg = tasks[task] or {}
            if "backbone" in tcfg:
                flat["backbone"] = tcfg["backbone"]
            for key in ("lr", "iterations", "gamma", "omega", "J"):
                if key in tcfg:
                    flat.setdefault(key, tcfg[key])

        # flat-style config
        for key in list(flat.keys()):
            pass
        known = set(SamplingConfig.__dataclass_fields__.keys())
        for key, val in cfg.items():
            if key in known:
                flat[key] = val

        flat["backbone"] = overrides.pop("backbone", flat.get("backbone", backbone))
        flat.update(overrides)
        flat = {k: v for k, v in flat.items() if k in known}
        return SamplingConfig(**flat)

    def replace(self, **overrides) -> "SamplingConfig":
        data = {k: getattr(self, k) for k in self.__dataclass_fields__}
        data.update({k: v for k, v in overrides.items() if k in self.__dataclass_fields__})
        return SamplingConfig(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @property
    def effective_eta(self) -> float:
        if self.eta is not None:
            return float(self.eta)
        return 0.0 if str(self.method).lower().startswith("ddim") else 1.0

    def timestep_sequence(self, num_steps: Optional[int] = None) -> torch.Tensor:
        """Descending timestep subsequence for accelerated DDIM sampling.

        For the full DDPM chain (num_steps >= T or eta == 1 with method ddpm) this
        returns ``[T, T-1, ..., 1]``; otherwise it is the uniform ``num_steps``
        strided subset of ``[1, T]`` (state-space style accelerated sampling).
        """
        T = int(self.num_timesteps)
        steps = int(num_steps if num_steps is not None else self.num_steps)
        if self.method.lower().startswith("ddpm") and self.eta is None:
            steps = T
        steps = max(1, min(steps, T))
        if steps >= T:
            return torch.arange(T, 0, -1, dtype=torch.long)
        # uniform strided subsequence including T and (approximately) 1
        ts = torch.linspace(T, 1, steps + 1).round().long()
        # deduplicate while preserving descending order
        uniq: List[int] = []
        for v in ts.tolist():
            if not uniq or uniq[-1] != v:
                uniq.append(int(v))
        return torch.tensor(uniq, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #
class Sampler:
    """Generic ancestor sampler implementing Eq. (2)-(4) for adapted DPMs."""

    def __init__(
        self,
        model: Any,
        classifier: Optional[Any] = None,
        config: Optional[Union[SamplingConfig, Dict[str, Any]]] = None,
        schedule: Optional[NoiseSchedule] = None,
        autoencoder: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        backbone: Optional[str] = None,
        **overrides,
    ):
        if isinstance(config, dict):
            config = SamplingConfig.from_dict(config, backbone=backbone or "ddpm", **overrides)
        elif config is None:
            config = SamplingConfig.from_dict(None, backbone=backbone or "ddpm", **overrides)
        elif overrides:
            config = config.replace(**overrides)
        self.config = config

        if device is None:
            device = config.device if torch.cuda.is_available() or str(config.device) == "cpu" else "cpu"
        self.device = torch.device(device)
        self.dtype = getattr(torch, str(config.dtype), torch.float32)

        self.model = model
        self.classifier = classifier
        self.autoencoder = autoencoder
        self.backbone = (backbone or config.backbone or "ddpm").lower()
        self.predict_noise = _resolve_callable(model)

        if schedule is None:
            schedule = build_schedule(
                None,
                num_timesteps=config.num_timesteps,
                schedule=config.schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                cosine_s=config.cosine_s,
            )
        self.schedule = schedule.to(self.device)

    # -- schedule accessors ------------------------------------------------ #
    @property
    def num_timesteps(self) -> int:
        return int(self.config.num_timesteps)

    def _ab(self, t: torch.Tensor) -> torch.Tensor:
        return _extract(self.schedule.alphas_cumprod, t, 4)

    def _ab_prev(self, t: torch.Tensor) -> torch.Tensor:
        return _extract(self.schedule.alphas_cumprod_prev, t, 4)

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return _extract(self.schedule.alphas, t, 4)

    def sigma_hat(self, t: torch.Tensor) -> torch.Tensor:
        """Appendix A.2 coefficient sigma_hat_t, used in Eq. (4)."""
        return _extract(self.schedule.sigma_hat, t, 4)

    def reverse_sigma(self, t: torch.Tensor, eta: float) -> torch.Tensor:
        """sigma_t from Eq. (3): 0 for DDIM (eta=0), DDPM value for eta=1."""
        ab = self._ab(t)
        ab_prev = self._ab_prev(t)
        if eta == 0.0:
            return torch.zeros_like(ab)
        var = (1.0 - ab_prev) / (1.0 - ab).clamp(min=1e-20) * (1.0 - ab / ab_prev.clamp(min=1e-20))
        var = var.clamp(min=0.0)
        return float(eta) * var.sqrt()

    # -- model calls -------------------------------------------------------- #
    def predict_eps(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.predict_noise(x_t, t)
        out = _unwrap_prediction(out)
        # Some wrappers return concatenated [eps, sigma] when learn_sigma=True
        if out.shape[1] % 2 == 0 and out.shape[1] == x_t.shape[1] * 2:
            out = out[:, : x_t.shape[1]]
        return out

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eps is None:
            eps = self.predict_eps(x_t, t)
        ab = self._ab(t)
        x0 = (x_t - (1.0 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-20)
        if self.config.clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)
        return x0

    def classifier_gradient(self, x_t: torch.Tensor, t: torch.Tensor,
                            target_index: Optional[int] = None) -> Optional[torch.Tensor]:
        """Detached ∇_{x_t} log p_phi(y=T|x_t) (Eq. 4)."""
        if self.classifier is None or not self.config.use_classifier_guidance:
            return None
        idx = self.config.classifier_target_index if target_index is None else target_index
        with torch.enable_grad():
            x = x_t.detach().requires_grad_(True)
            grad = None
            fn = getattr(self.classifier, "grad_log_target", None)
            if callable(fn):
                try:
                    grad = fn(x, t, target_index=idx, detach=True)
                except TypeError:
                    try:
                        grad = fn(x, t, detach=True)
                    except TypeError:
                        grad = fn(x, t)
            else:
                out = self.classifier(x, t)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                logp = F.log_softmax(out, dim=1)[:, idx].sum()
                grad = torch.autograd.grad(logp, x, create_graph=False)[0]
        if grad is None:
            return None
        return grad.detach()

    def guided_eps(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Eq. (4): eps_theta(x_t,t) - sigma_hat_t^2 * gamma * grad log p_phi."""
        eps = self.predict_eps(x_t, t)
        grad = self.classifier_gradient(x_t, t)
        if grad is not None:
            gamma = float(self.config.guidance_scale)
            eps = eps - self.sigma_hat(t) ** 2 * gamma * grad
        return eps

    # -- reverse step ------------------------------------------------------- #
    @torch.no_grad()
    def reverse_step(self, x_t: torch.Tensor, t: int, t_prev: int,
                     eta: Optional[float] = None, add_noise: bool = True) -> torch.Tensor:
        """One reverse step following Eq. (2)/(3) using the adapted model."""
        eta = self.config.effective_eta if eta is None else float(eta)
        b = x_t.shape[0]
        tt = torch.full((b,), int(t), device=self.device, dtype=torch.long)

        if self.config.use_classifier_guidance and self.classifier is not None:
            eps = self.guided_eps(x_t, tt)
        else:
            eps = self.predict_eps(x_t, tt)

        ab_t = self._ab(tt)
        # target alpha_bar for the *previous* (possibly skipped) timestep
        if t_prev is None or t_prev <= 0:
            ab_prev = torch.ones_like(ab_t)
        else:
            ab_prev = _extract(
                self.schedule.alphas_cumprod,
                torch.full((b,), int(t_prev), device=self.device, dtype=torch.long),
                4,
            )

        sigma = self.reverse_sigma(tt, eta)
        # x0 estimate from eps
        x0 = (x_t - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-20)
        if self.config.clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)

        # Eq. (3): deterministic direction with the (possibly partial) sigma_t
        dir_coef = (1.0 - ab_prev - sigma ** 2).clamp(min=0.0).sqrt()
        x_prev = ab_prev.sqrt() * x0 + dir_coef * eps

        if float(eta) > 0.0 and add_noise and int(t_prev) > 0:
            x_prev = x_prev + sigma * torch.randn_like(x_prev)
        return x_prev

    # -- full sampling loops ------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        batch_size: Optional[int] = None,
        num_samples: Optional[int] = None,
        shape: Optional[Sequence[int]] = None,
        num_steps: Optional[int] = None,
        eta: Optional[float] = None,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        decode: bool = True,
        verbose: bool = False,
        return_latents: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples with the adapted model.

        Returns a tensor in image space ``(N, 3, H, W)`` (decoded via the frozen
        autoencoder for LDM backbones) or, when ``return_latents``/``decode=False``,
        the raw latent tensor.
        """
        cfg = self.config
        bs = int(batch_size or cfg.batch_size)
        total = int(num_samples or cfg.num_samples)
        steps = int(num_steps or cfg.num_steps)
        eta = cfg.effective_eta if eta is None else float(eta)

        sample_shape = tuple(shape) if shape is not None else self._default_shape()
        latent_shape = (bs,) + tuple(sample_shape)

        # timestep subsequence (descending)
        ts = cfg.timestep_sequence(steps).to(self.device)
        if verbose:
            LOGGER.info("Sampling %d images in batches of %d with %d steps (eta=%.2f, backbone=%s)",
                        total, bs, len(ts), eta, self.backbone)

        if seed is not None and generator is None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            self._generator_device = "cpu"
        else:
            self._generator_device = None

        outputs: List[torch.Tensor] = []
        remaining = total
        while remaining > 0:
            n = min(bs, remaining)
            x = self._init_latents((n,) + tuple(sample_shape), generator, latents)
            for i, t in enumerate(ts.tolist()):
                t_prev = int(ts[i + 1]) if i + 1 < len(ts) else 0
                x = self.reverse_step(x, t, t_prev, eta=eta)
            outputs.append(x.detach().float().cpu())
            remaining -= n

        latents_all = torch.cat(outputs, dim=0)[:total]
        latents_all = latents_all.to(self.device)

        if self.backbone == "ldm" and decode and self.autoencoder is not None:
            img = self.decode(latents_all)
            if return_latents:
                return img, latents_all.detach().float().cpu()
            return img
        if return_latents:
            return latents_all, latents_all.detach().float().cpu()
        return latents_all

    # -- utilities ---------------------------------------------------------- #
    def _default_shape(self) -> Tuple[int, ...]:
        if self.backbone == "ldm":
            latent_size = getattr(self.autoencoder, "latent_size", None)
            if latent_size is None:
                latent_size = 64
            ch = getattr(self.autoencoder, "latent_channels", None) or 4
            return (int(ch), int(latent_size), int(latent_size))
        # pixel-space DDPM: infer from model config if available
        model = self.model
        base = getattr(model, "unet", model)
        in_ch = getattr(base, "in_channels", None)
        if in_ch is None:
            in_ch = 3
        img_size = getattr(base, "image_size", 256) or 256
        if isinstance(img_size, (tuple, list)):
            img_size = img_size[0]
        return (int(in_ch), int(img_size), int(img_size))

    def _init_latents(self, shape: Tuple[int, ...],
                      generator: Optional[torch.Generator] = None,
                      latents: Optional[torch.Tensor] = None) -> torch.Tensor:
        if latents is not None:
            return latents.to(self.device).to(self.dtype).clone()
        if generator is None:
            return torch.randn(shape, device=self.device, dtype=self.dtype)
        # move the generator to the sampling device when supported
        gen = generator
        try:
            gen = torch.Generator(device=self.device).manual_seed(
                int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator).item())
            )
        except Exception:  # pragma: no cover - device-specific generator support
            gen = generator
        return torch.randn(shape, device=self.device, dtype=self.dtype, generator=gen)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel space with the frozen autoencoder."""
        if self.autoencoder is None:
            return latents
        fn = getattr(self.autoencoder, "decode", None)
        if callable(fn):
            try:
                return fn(latents)
            except TypeError:
                pass
        if callable(self.autoencoder):
            return self.autoencoder(latents)
        return latents

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.autoencoder is None:
            return images
        fn = getattr(self.autoencoder, "encode", None)
        if callable(fn):
            return fn(images)
        return images

    def sample_to_dir(self, out_dir: str, num_samples: Optional[int] = None,
                      batch_size: Optional[int] = None, num_steps: Optional[int] = None,
                      eta: Optional[float] = None, seed: Optional[int] = None,
                      prefix: str = "sample", ext: str = "png",
                      to_uint8: bool = True, verbose: bool = True) -> List[str]:
        """Generate images and save them individually (convenience for evaluation)."""
        os.makedirs(out_dir, exist_ok=True)
        imgs = self.sample(batch_size=batch_size, num_samples=num_samples,
                           num_steps=num_steps, eta=eta, seed=seed, verbose=verbose)
        if isinstance(imgs, tuple):
            imgs = imgs[0]
        paths: List[str] = []
        try:
            from PIL import Image
            have_pil = True
        except Exception:  # pragma: no cover
            have_pil = False
        for i in range(imgs.shape[0]):
            arr = imgs[i].detach().cpu().float()
            if to_uint8:
                arr = ((arr.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
            path = os.path.join(out_dir, f"{prefix}_{i:06d}.{ext}")
            if have_pil:
                np_arr = arr.permute(1, 2, 0).numpy()
                Image.fromarray(np_arr).save(path)
            else:  # pragma: no cover
                torch.save(arr, path)
            paths.append(path)
        return paths


class DDIMSampler(Sampler):
    """Deterministic (eta = 0) DDIM sampler - default for evaluation (Section 5.2)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("method", "ddim")
        kwargs.setdefault("eta", 0.0)
        super().__init__(*args, **kwargs)
        self.config.method = "ddim"
        self.config.eta = 0.0


# --------------------------------------------------------------------------- #
# Factories and functional wrappers
# --------------------------------------------------------------------------- #
def build_sampler(
    cfg: Optional[Dict[str, Any]] = None,
    model: Any = None,
    classifier: Optional[Any] = None,
    autoencoder: Optional[Any] = None,
    backbone: str = "ddpm",
    task: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    schedule: Optional[NoiseSchedule] = None,
    **overrides,
) -> Sampler:
    """Config-driven sampler factory."""
    config = SamplingConfig.from_dict(cfg, backbone=backbone, task=task, **overrides)
    if device is None:
        device = config.device
    return Sampler(
        model=model,
        classifier=classifier,
        config=config,
        schedule=schedule,
        autoencoder=autoencoder,
        device=device,
        backbone=config.backbone,
    )


@torch.no_grad()
def ddim_sample(model, num_samples: int = 1000, num_steps: int = 100,
                batch_size: int = 16, etas: float = 0.0, shape=None,
                device=None, schedule=None, num_timesteps: int = 1000,
                schedule_name: str = "linear", classifier=None,
                guidance_scale: float = 5.0, seed: Optional[int] = None,
                autoencoder=None, backbone: str = "ddpm") -> torch.Tensor:
    """Functional DDIM sampling (Eq. 2/3 with eta=0)."""
    cfg = SamplingConfig(method="ddim", num_steps=num_steps, eta=etas,
                         batch_size=batch_size, num_samples=num_samples,
                         num_timesteps=num_timesteps, schedule=schedule_name,
                         device=str(device) if device is not None else "cuda",
                         use_classifier_guidance=classifier is not None,
                         guidance_scale=guidance_scale, backbone=backbone)
    sampler = Sampler(model, classifier=classifier, config=cfg, schedule=schedule,
                      autoencoder=autoencoder, device=device, backbone=backbone)
    return sampler.sample(batch_size=batch_size, num_samples=num_samples,
                          shape=shape, num_steps=num_steps, eta=etas, seed=seed)


@torch.no_grad()
def ddpm_sample(model, num_samples: int = 1000, batch_size: int = 16,
                shape=None, device=None, schedule=None, num_timesteps: int = 1000,
                schedule_name: str = "linear", classifier=None,
                guidance_scale: float = 5.0, seed: Optional[int] = None,
                autoencoder=None, backbone: str = "ddpm") -> torch.Tensor:
    """Functional DDPM sampling: full T-step stochastic reverse process (Eq. 2)."""
    cfg = SamplingConfig(method="ddpm", num_steps=num_timesteps, eta=1.0,
                         batch_size=batch_size, num_samples=num_samples,
                         num_timesteps=num_timesteps, schedule=schedule_name,
                         device=str(device) if device is not None else "cuda",
                         use_classifier_guidance=classifier is not None,
                         guidance_scale=guidance_scale, backbone=backbone)
    sampler = Sampler(model, classifier=classifier, config=cfg, schedule=schedule,
                      autoencoder=autoencoder, device=device, backbone=backbone)
    return sampler.sample(batch_size=batch_size, num_samples=num_samples,
                          shape=shape, num_steps=num_timesteps, eta=1.0, seed=seed)


def sample_images(model, cfg: Optional[Dict[str, Any]] = None,
                  classifier: Optional[Any] = None, autoencoder: Optional[Any] = None,
                  backbone: str = "ddpm", task: Optional[str] = None,
                  device: Optional[Union[str, torch.device]] = None,
                  out_dir: Optional[str] = None,
                  num_samples: Optional[int] = None,
                  num_steps: Optional[int] = None,
                  seed: Optional[int] = None,
                  **overrides) -> torch.Tensor:
    """High-level entry point used by ``scripts/sample.py`` and evaluation."""
    sampler = build_sampler(cfg=cfg, model=model, classifier=classifier,
                            autoencoder=autoencoder, backbone=backbone, task=task,
                            device=device, **overrides)
    if out_dir is not None:
        sampler.sample_to_dir(out_dir, num_samples=num_samples,
                              num_steps=num_steps, seed=seed)
        return sampler  # type: ignore[return-value]
    return sampler.sample(num_samples=num_samples, num_steps=num_steps, seed=seed)
