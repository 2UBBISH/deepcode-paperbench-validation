#!/usr/bin/env python3
"""CPU smoke test of the *complete* DPMs-ANT pipeline on synthetic data.

It runs the same code path as the real experiments (adaptor injection,
classifier fine-tuning on noised images, adversarial noise selection,
similarity-guided training, sampling and Intra-LPIPS) but with a tiny U-Net
and a handful of iterations, so that it finishes in a couple of minutes
without needing the paper's datasets or checkpoints.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import TensorDataset

from dpms_ant.adaptor import AdaptorConfig, add_adaptors
from dpms_ant.ant_trainer import ANTConfig, ANTTrainer, eps_predictor
from dpms_ant.backbones import (
    Classifier256Config,
    DDPM256Config,
    build_classifier_unet,
    build_ddpm_unet,
    replace_classifier_head,
)
from dpms_ant.classifier import ClassifierTrainConfig, train_domain_classifier
from dpms_ant.data import LabelledImageDataset, infinite_loader
from dpms_ant.guidance import similarity_guided_loss
from dpms_ant.metrics import LPIPSMetric, intra_lpips
from dpms_ant.sampling import sample_loop
from dpms_ant.schedules import DiffusionSchedule, respaced_timesteps
from dpms_ant.utils import Logger, parameter_rate, set_seed


def synthetic_dataset(count: int, image_size: int, colour: float, seed: int) -> TensorDataset:
    """A synthetic "domain": coloured blobs (source = red-ish, target = blue-ish)."""
    generator = torch.Generator().manual_seed(seed)
    images = torch.zeros(count, 3, image_size, image_size)
    for index in range(count):
        images[index, 0] = colour
        images[index, 2] = 1.0 - colour
        centre = torch.randint(4, image_size - 4, (2,), generator=generator)
        size = int(torch.randint(4, 9, (1,), generator=generator))
        images[
            index,
            :,
            centre[0] : centre[0] + size,
            centre[1] : centre[1] + size,
        ] = torch.randn(3, 1, 1, generator=generator) * 0.5
    return TensorDataset(images.clamp(-1, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--source-iterations", type=int, default=30)
    parser.add_argument("--classifier-iterations", type=int, default=30)
    parser.add_argument("--num-generated", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(0)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, "log.txt"))

    schedule = DiffusionSchedule().to(args.device)
    model = build_ddpm_unet(
        DDPM256Config(
            image_size=args.image_size,
            num_channels=32,
            num_res_blocks=1,
            channel_mult="1,2",
            attention_resolutions="8,4",
            num_head_channels=16,
        )
    ).to(args.device)
    model.convert_to_fp32()
    logger.log(f"[smoke] U-Net parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # --- "pre-train" the source model (a few steps is enough for the smoke test)
    source = synthetic_dataset(64, args.image_size, colour=1.0, seed=0)
    target = synthetic_dataset(10, args.image_size, colour=-1.0, seed=1)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(0)
    model.train()
    for step in range(args.source_iterations):
        x0 = torch.stack([source[i][0] for i in torch.randint(0, 64, (4,), generator=generator)])
        t = torch.randint(1, schedule.num_timesteps, (4,), generator=generator).long()
        noise = torch.randn(x0.shape, generator=generator)
        loss = similarity_guided_loss(model(schedule.q_sample(x0, t, noise), t)[:, :3], noise)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    logger.log(f"[smoke] source pre-training loss {float(loss):.4f}")

    # --- adaptors (only psi is trainable) ------------------------------
    add_adaptors(
        model,
        AdaptorConfig(bottleneck_factor=4, bottleneck_dim=8),
        example_input=torch.randn(1, 3, args.image_size, args.image_size, device=args.device),
        forward_kwargs={"timesteps": torch.zeros(1, dtype=torch.long, device=args.device)},
    )
    logger.log(f"[smoke] adaptor parameter rate {100 * parameter_rate(model):.2f}%")

    # --- domain classifier on noised images ----------------------------
    clf = build_classifier_unet(
        num_classes=2, config=Classifier256Config(image_size=args.image_size)
    )
    replace_classifier_head(clf, num_classes=2)
    labelled = LabelledImageDataset(source, target)
    train_domain_classifier(
        clf,
        schedule,
        infinite_loader(labelled, batch_size=8, shuffle=True),
        ClassifierTrainConfig(iterations=args.classifier_iterations, lr=1e-4, batch_size=8),
        device=args.device,
        logger=logger,
    )

    # --- Algorithm 1 ---------------------------------------------------
    config = ANTConfig(
        iterations=args.iterations,
        batch_size=10,
        lr=5e-5,
        gamma=5.0,
        adaptor=AdaptorConfig(bottleneck_factor=4, bottleneck_dim=8),
    )
    config.adversarial.num_steps = 3
    config.log_every = 1
    trainer = ANTTrainer(
        model, clf, schedule, config, device=args.device, logger=logger, in_channels=3
    )
    history = trainer.train(infinite_loader(target, batch_size=config.batch_size))
    logger.log(
        f"[smoke] ANT loss {history['loss'][0]:.4f} -> {history['loss'][-1]:.4f}"
    )

    # --- sampling + Intra-LPIPS ---------------------------------------
    steps = torch.flip(
        respaced_timesteps(schedule.num_timesteps, "ddim10", device=args.device), dims=[0]
    )
    with torch.no_grad():
        samples = sample_loop(
            schedule,
            (args.num_generated, 3, args.image_size, args.image_size),
            eps_predictor(model, 3),
            steps=steps,
            eta=0.0,
            device=args.device,
        )
    target_images = torch.stack([target[i][0] for i in range(len(target))])
    score = intra_lpips(samples, target_images, metric=LPIPSMetric(device=args.device), device=args.device)
    logger.log(f"[smoke] sampled {tuple(samples.shape)}, Intra-LPIPS = {score:.4f}")
    logger.log("[smoke] OK")


if __name__ == "__main__":
    main()
