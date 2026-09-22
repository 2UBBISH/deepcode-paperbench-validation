"""Sampling with coupled stochastic interpolants (Algorithm 2 / Corollary 3.1).

    Algorithm 2 Sampling (via forward Euler method)
        Input: model b_hat, corrupted sample m(x_1), N in N
        Draw noise zeta ~ N(0, Id)
        Initialise X_0 = m(x_1) + sigma zeta
        for n = 0, ..., N - 1 do
            X_{i+1} = X_i + N^{-1} b_hat_{i/N}(X_i)
        end for
        Return: clean sample X_N

The paper integrates the probability-flow ODE with the Dopri solver of
``torchdiffeq`` (Appendix B); ``method="dopri5"`` reproduces that choice with
the self-contained solver of :mod:`si_couplings.solvers`, and
``method="euler"`` follows Algorithm 2 literally.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from .couplings import Coupling, InpaintingCoupling, SuperResolutionCoupling
from .interpolants import InterpolantSchedule
from .solvers import odeint
from .utils import load_checkpoint, resolve_device


def make_velocity_fn(model, cond: Optional[Tensor], labels: Optional[Tensor], mask: Optional[Tensor],
                     guidance_scale: Optional[float] = None):
    """Wrap the model into a function v(x, t) suitable for the ODE/SDE solvers."""

    def velocity_fn(x: Tensor, t: Tensor) -> Tensor:
        if guidance_scale is not None and hasattr(model, "guided_velocity"):
            return model.guided_velocity(x, t, cond, labels, mask, guidance_scale=guidance_scale)
        return model(x, t, cond, labels, mask)

    return velocity_fn


@torch.no_grad()
def sample_from_base(
    model,
    x0: Tensor,
    *,
    cond: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    method: str = "dopri5",
    steps: int = 250,
    t0: float = 0.0,
    t1: float = 1.0,
    guidance_scale: Optional[float] = None,
    **solver_kwargs,
) -> Tensor:
    """Integrate the probability-flow ODE from x0 at t0 to t1."""
    velocity_fn = make_velocity_fn(model, cond, labels, mask, guidance_scale)
    return odeint(velocity_fn, x0, t0=t0, t1=t1, method=method, steps=steps, **solver_kwargs)


@torch.no_grad()
def sample_inpainting(
    model,
    coupling: Coupling,
    ground_truth: Tensor,
    *,
    xi: Optional[Tensor] = None,
    mask_spec: Optional[tuple] = None,
    labels: Optional[Tensor] = None,
    method: str = "dopri5",
    steps: int = 250,
    generator: Optional[torch.Generator] = None,
    guidance_scale: Optional[float] = None,
) -> dict:
    """In-paint a batch of images (Section 4.1).

    ``coupling`` should normally be an
    :class:`~si_couplings.couplings.InpaintingCoupling`, in which case the
    base sample is x_0 = xi o x_1 + (1 - xi) o zeta.  Any other coupling can
    be used as the uncoupled baseline: the *same* mask is drawn, but the base
    sample is drawn from that coupling instead (e.g. x_0 ~ N(0, Id) for
    :class:`~si_couplings.couplings.IndependentCoupling`), which reproduces
    the baseline of Table 2.

    Returns a dict with the masked base sample ``x0``, the model sample
    ``sample`` and the ground truth ``x1``, together with the mask ``xi``.
    """
    b, c, h, w = ground_truth.shape
    from .couplings import InpaintingCoupling, tile_mask

    if mask_spec is None:
        n_tiles = getattr(coupling, "n_tiles", 8)
        p_missing = getattr(coupling, "p_missing", 0.3)
    else:
        n_tiles, p_missing = mask_spec
    if xi is None:
        xi = tile_mask(b, c, h, w, n_tiles, p_missing, ground_truth.device, ground_truth.dtype)
    xi_full = xi.expand(b, c, h, w) if xi.shape[1] == 1 else xi
    mask = 1.0 - xi_full
    if isinstance(coupling, InpaintingCoupling):
        zeta = torch.randn(ground_truth.shape, device=ground_truth.device, dtype=ground_truth.dtype,
                           generator=generator)
        x0 = xi_full * ground_truth + mask * zeta
        cond = xi_full
    else:
        batch = coupling.sample(ground_truth, labels=labels)
        x0, cond = batch.x0, batch.cond
        if coupling.masks_output and batch.mask is not None:
            mask = batch.mask
    sample = sample_from_base(model, x0, cond=cond, labels=labels, mask=mask,
                              method=method, steps=steps, guidance_scale=guidance_scale)
    return {"x0": x0, "sample": sample, "x1": ground_truth, "xi": xi_full, "mask": mask}


@torch.no_grad()
def sample_super_resolution(
    model,
    coupling: SuperResolutionCoupling,
    low_res: Tensor,
    *,
    labels: Optional[Tensor] = None,
    sigma: Optional[float] = None,
    method: str = "dopri5",
    steps: int = 250,
    generator: Optional[torch.Generator] = None,
    guidance_scale: Optional[float] = None,
    high_res_truth: Optional[Tensor] = None,
) -> dict:
    """Super-resolve a batch of low-resolution images (Section 4.2).

    ``low_res`` is the low-resolution image; it is up-sampled to the target
    resolution and used both as the mean of the base density and as the
    conditioning signal xi = U(D(x_1)).
    """
    sigma = coupling.sigma if sigma is None else sigma
    target_size = (
        high_res_truth.shape[-2:] if high_res_truth is not None
        else (low_res.shape[-2] * coupling.scale, low_res.shape[-1] * coupling.scale)
    )
    xi = coupling.upsample(low_res, size=target_size)
    # the coupling defines how the base sample is built from xi: the coupled
    # one uses xi + sigma zeta, the uncoupled baseline uses N(0, Id)
    if hasattr(coupling, "base_from_xi") and sigma == getattr(coupling, "sigma", None):
        x0 = coupling.base_from_xi(xi, generator=generator)
    else:
        x0 = xi + sigma * torch.randn(xi.shape, device=xi.device, dtype=xi.dtype,
                                      generator=generator)
    sample = sample_from_base(model, x0, cond=xi, labels=labels, mask=None,
                              method=method, steps=steps, guidance_scale=guidance_scale)
    return {"x0": x0, "sample": sample, "xi": xi, "low_res": low_res, "x1": high_res_truth}


@torch.no_grad()
def sample_unconditional(
    model,
    shape,
    *,
    labels: Optional[Tensor] = None,
    method: str = "dopri5",
    steps: int = 250,
    device=None,
    base_scale: float = 1.0,
    guidance_scale: Optional[float] = None,
) -> Tensor:
    """Sample from the target through an independent Gaussian base (baseline)."""
    device = resolve_device("auto") if device is None else device
    x0 = torch.randn(shape, device=device) * base_scale
    return sample_from_base(model, x0, labels=labels, method=method, steps=steps,
                            guidance_scale=guidance_scale)


def load_model_from_checkpoint(cfg: dict, checkpoint: str | os.PathLike, use_ema: bool = True):
    """Rebuild the velocity model of a run and load its weights."""
    from .train import build_experiment

    parts = build_experiment(cfg)
    model = parts["model"]
    payload = load_checkpoint(checkpoint, map_location=parts["device"])
    state = payload["ema"] if (use_ema and payload.get("ema") is not None) else payload["model"]
    model.load_state_dict(state)
    model.eval()
    return model, parts["coupling"], parts["schedule"], parts["device"]


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sample from a trained interpolant.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default=None, choices=[None, "inpainting", "super_resolution", "unconditional"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--out", default="results/samples.pt")
    args = parser.parse_args(argv)

    from .utils import load_config

    cfg = load_config(args.config)
    task = args.task or cfg.get("task", "inpainting")
    sampling = cfg.get("sampling", {})
    method = args.method or sampling.get("method", "dopri5")
    steps = int(args.steps or sampling.get("steps", 250))

    model, coupling, schedule, device = load_model_from_checkpoint(cfg, args.checkpoint)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if task == "super_resolution":
        from .data.superres import downsample

        high = torch.rand(args.n, 3, 256, 256, device=device) * 2 - 1
        low = downsample(high, coupling.scale)
        result = sample_super_resolution(model, coupling, low, method=method, steps=steps,
                                         high_res_truth=high)
    elif task == "unconditional":
        result = {"sample": sample_unconditional(model, (args.n, 3, 256, 256), method=method,
                                                 steps=steps, device=device)}
    else:
        images = torch.rand(args.n, 3, 256, 256, device=device) * 2 - 1
        labels = torch.randint(0, 1000, (args.n,), device=device)
        result = sample_inpainting(model, coupling, images, labels=labels, method=method, steps=steps)
    torch.save(result, out)
    print(f"wrote {out}")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "sample_from_base",
    "sample_inpainting",
    "sample_super_resolution",
    "sample_unconditional",
    "load_model_from_checkpoint",
    "main",
]
