"""Training loop (Algorithm 1) for stochastic interpolants with couplings.

    Algorithm 1 Training
        Input: interpolant coefficients alpha_t, beta_t; velocity model b_hat;
               batch size n_b;
        repeat
            for i = 1, ..., n_b do
                Draw x_1^i ~ rho_1, zeta_i ~ N(0, Id), t_i ~ U(0, 1)
                Compute x_0^i = m(x_1^i) + sigma zeta^i
                Compute I_{t_i} = alpha_{t_i} x_0^i + beta_{t_i} x_1^i
            end for
            Compute empirical loss
                L_hat_b(b_hat) = 1/n_b sum_i [ |b_hat_{t_i}(I_{t_i})|^2
                                               - 2 I_dot_{t_i} . b_hat_{t_i}(I_{t_i}) ]
            Take gradient step on L_hat_b(b_hat) to update b_hat
        until converged

Optimisation hyper-parameters of Appendix B: Adam starting at learning rate
2e-4 with a StepLR scheduler scaling the learning rate by gamma = 0.99 every
N = 1000 steps, no weight decay, gradient-norm clipping at 10,000, batch size
32 and 200,000 gradient steps (the last two are stated in the addendum).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .couplings import Coupling, build_coupling
from .interpolants import InterpolantSchedule, build_interpolant
from .losses import velocity_loss
from .models.velocity import build_velocity_model
from .utils import EMA, JsonlLogger, count_parameters, resolve_device, save_checkpoint, set_seed


# ---------------------------------------------------------------------------
def build_experiment(cfg: dict, device: Optional[torch.device] = None) -> dict:
    """Instantiate every component of an experiment from its configuration."""
    device = resolve_device(cfg.get("device", "auto")) if device is None else device
    schedule: InterpolantSchedule = build_interpolant(cfg.get("interpolant", "linear_zero_gamma"))
    coupling: Coupling = build_coupling(
        cfg.get("coupling", {}).get("name", "independent"),
        **{k: v for k, v in cfg.get("coupling", {}).items() if k != "name"},
    )
    model = build_velocity_model(cfg, coupling.info).to(device)
    return {"schedule": schedule, "coupling": coupling, "model": model, "device": device}


def build_optimizer(model: torch.nn.Module, cfg: dict):
    optim_cfg = cfg.get("optim", {})
    lr = float(optim_cfg.get("lr", 2e-4))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
        eps=float(optim_cfg.get("eps", 1e-8)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(optim_cfg.get("lr_decay_every", 1000)),
        gamma=float(optim_cfg.get("lr_decay_gamma", 0.99)),
    )
    return optimizer, scheduler


# ---------------------------------------------------------------------------
def train(
    cfg: dict,
    *,
    out_dir: str | os.PathLike | None = None,
    max_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    dataloader: Optional[DataLoader] = None,
) -> Path:
    """Run Algorithm 1 and return the path of the last checkpoint."""
    set_seed(int(cfg.get("seed", 0)))
    parts = build_experiment(cfg, device)
    model, schedule, coupling, device = parts["model"], parts["schedule"], parts["coupling"], parts["device"]

    out_dir = Path(out_dir or cfg.get("out_dir", "runs/experiment"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    logger = JsonlLogger(out_dir / "train.jsonl")
    logger.log(params=count_parameters(model), coupling=coupling.name, interpolant=schedule.name)

    if dataloader is None:
        from .data.imagenet import build_dataloader

        dataloader = build_dataloader(cfg, train=True)

    optimizer, scheduler = build_optimizer(model, cfg)
    ema = EMA(model, beta=float(cfg.get("optim", {}).get("ema_beta", 0.995))) if cfg.get("optim", {}).get(
        "ema", True
    ) else None

    optim_cfg = cfg.get("optim", {})
    max_steps = int(max_steps or optim_cfg.get("max_steps", 200_000))
    grad_clip = float(optim_cfg.get("grad_clip_norm", 10_000.0))
    log_every = int(optim_cfg.get("log_every", 50))
    save_every = int(optim_cfg.get("save_every", 5_000))
    label_dropout = float(optim_cfg.get("label_dropout", 0.0))
    mse_form = bool(optim_cfg.get("mse_form", True))

    step = 0
    t0 = time.time()
    model.train()
    while step < max_steps:
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if label_dropout > 0 and getattr(model.backbone, "num_classes", None) is not None:
                drop = torch.rand(labels.shape, device=device) < label_dropout
                labels = torch.where(drop, torch.full_like(labels, model.backbone.null_class), labels)

            batch = coupling.sample(images, labels=labels)
            loss, info = velocity_loss(model, schedule, batch, mse_form=mse_form)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            step += 1

            if step % log_every == 0 or step == 1:
                logger.log(
                    step=step,
                    loss=float(loss.detach()),
                    idot_sq=float(info["idot_sq"]),
                    lr=optimizer.param_groups[0]["lr"],
                    steps_per_s=(step / max(time.time() - t0, 1e-6)),
                )
            if step % save_every == 0 or step == max_steps:
                ckpt = out_dir / f"checkpoint_{step:07d}.pt"
                save_checkpoint(ckpt, model, ema.model if ema is not None else None,
                                optimizer, step, cfg)
                save_checkpoint(out_dir / "checkpoint_last.pt", model,
                                ema.model if ema is not None else None, optimizer, step, cfg)
            if step >= max_steps:
                break
    return out_dir / "checkpoint_last.pt"


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a stochastic interpolant with couplings.")
    parser.add_argument("--config", required=True, help="path to a YAML/JSON configuration")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="override optim.max_steps")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    from .utils import load_config

    cfg = load_config(args.config)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    train(cfg, out_dir=args.out_dir, max_steps=args.max_steps,
          device=None if args.device is None else torch.device(args.device))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["train", "build_experiment", "build_optimizer", "main"]
