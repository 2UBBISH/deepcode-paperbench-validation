#!/usr/bin/env python3
"""Sample images from a (transferred) DDPM/LDM checkpoint.

python scripts/generate.py --checkpoint checkpoints/256x256_diffusion_uncond.pt \
    --adaptor outputs/ddpm_ffhq_sunglasses/checkpoints/adaptor.pt \
    --num-samples 64 --output outputs/samples.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from dpms_ant.adaptor import AdaptorConfig, add_adaptors
from dpms_ant.ant_trainer import eps_predictor
from dpms_ant.backbones import load_ddpm_256
from dpms_ant.sampling import sample_loop
from dpms_ant.schedules import DiffusionSchedule, respaced_timesteps
from dpms_ant.utils import get_device, save_image_grid, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--adaptor", default=None, help="payload saved by ANTTrainer")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--bottleneck-dim", type=int, default=8)
    parser.add_argument("--bottleneck-factor", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=0)
    parser.add_argument("--output", default="outputs/samples.png")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = str(get_device(args.device))
    set_seed(args.seed)
    model = load_ddpm_256(args.checkpoint).to(device)
    schedule = DiffusionSchedule().to(device)
    add_adaptors(
        model,
        AdaptorConfig(args.bottleneck_factor, args.bottleneck_dim),
        example_input=torch.randn(1, 3, args.image_size, args.image_size, device=device),
        forward_kwargs={"timesteps": torch.zeros(1, dtype=torch.long, device=device)},
    )
    if args.adaptor is not None:
        payload = torch.load(args.adaptor, map_location=device)
        state = payload.get("adaptor", payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[generate] loaded adaptor ({len(state)} tensors)")
    steps = None
    if args.ddim_steps:
        steps = torch.flip(
            respaced_timesteps(schedule.num_timesteps, f"ddim{args.ddim_steps}", device=device),
            dims=[0],
        )
    images = []
    for start in range(0, args.num_samples, args.batch_size):
        count = min(args.batch_size, args.num_samples - start)
        with torch.no_grad():
            images.append(
                sample_loop(
                    schedule,
                    (count, 3, args.image_size, args.image_size),
                    eps_predictor(model, 3),
                    steps=steps,
                    eta=0.0 if steps is not None else 1.0,
                    device=device,
                )
            )
    images = torch.cat(images)
    save_image_grid(images, args.output, nrow=min(8, args.num_samples))
    print(f"[generate] wrote {args.output}")


if __name__ == "__main__":
    main()

