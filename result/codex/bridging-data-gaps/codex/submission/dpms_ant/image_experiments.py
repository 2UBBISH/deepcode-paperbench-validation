"""End-to-end few-shot transfer experiments (Section 5.2 / 5.3, Tables 1-3).

For one ``(source, target, backbone)`` task this module

1. loads the pre-trained source model (guided-diffusion DDPM or an LDM) and the
   10-shot target dataset,
2. attaches zero-initialised adaptors and freezes the pre-trained weights
   (Section 4.3),
3. trains the binary source/target classifier on noised images
   (Section 5.2 + supplementary material),
4. runs Algorithm 1 (similarity-guided training + adversarial noise selection),
5. samples images, computes Intra-LPIPS / FID (Section 5.2) and writes the
   generated grids plus a ``results.json``.

The heavy lifting is done by the backbone-agnostic pieces of the package, so
DDPM and LDM only differ in how the model/latent space is provided.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .adaptor import add_adaptors
from .ant_trainer import ANTTrainer, eps_predictor, train_full_finetune
from .backbones import load_ddpm_256, load_domain_classifier, load_trained_domain_classifier
from .classifier import train_domain_classifier
from .configs import TaskConfig, get_task
from .data import (
    LabelledImageDataset,
    dataset_tensor,
    infinite_loader,
    list_images,
    target_dataset,
)
from .ldm_backend import LDMBackend, LDMConfig, PixelSpaceClassifier, build_ldm
from .metrics import FIDMetric, LPIPSMetric, compute_fid, intra_lpips
from .sampling import sample_loop
from .schedules import DiffusionSchedule
from .utils import Logger, ensure_dir, get_device, parameter_rate, save_image_grid, set_seed


@dataclass
class ExperimentArgs:
    """Command-line level configuration of one experiment."""

    task: str = "ddpm_ffhq_sunglasses"
    target_root: Optional[str] = None      # folder with the 10 target images
    source_root: Optional[str] = None      # folder with the source dataset
    checkpoint: Optional[str] = None       # pre-trained denoiser checkpoint
    classifier_checkpoint: Optional[str] = None
    output_dir: str = "outputs"
    device: str = "auto"
    seed: int = 0
    num_generated: int = 1000
    num_fid_images: int = 2500
    fid_real_root: Optional[str] = None
    baseline: str = "ant"                  # {"ant", "vanilla", "full"}
    classifier_freeze_backbone: bool = False
    adaptor_containers: Tuple[str, ...] = ("input_blocks", "middle_block", "output_blocks")
    adaptor_granularity: str = "layer"  # {"layer", "block"}
    ddim_steps: int = 0                    # 0 = full DDPM ancestral sampling
    save_every: int = 0


def build_backbone(
    task: TaskConfig, args: ExperimentArgs, device: str, logger: Optional[Logger] = None
) -> Tuple[nn.Module, DiffusionSchedule, int, bool]:
    """Return ``(eps_theta, schedule, in_channels, is_latent)``."""
    if task.backbone == "ddpm":
        model = load_ddpm_256(args.checkpoint)
        schedule = DiffusionSchedule(
            num_timesteps=1000, beta_start=1e-4, beta_end=2e-2, schedule="linear"
        )
        return model.to(device), schedule.to(device), 3, False

    if task.backbone == "ldm":
        ldm_config = LDMConfig(model_id=args.checkpoint or LDMConfig().model_id)
        if logger:
            logger.log(f"[ldm] loading {ldm_config.model_id}")
        backend, schedule = build_ldm(ldm_config, device=device)
        return backend, schedule, backend.latent_channels, True

    raise ValueError(f"unknown backbone {task.backbone!r}")


def encode_dataset_to_latents(backend: LDMBackend, dataset, device: str, batch: int = 8):
    """Pre-encode a small image dataset into the LDM latent space."""
    from torch.utils.data import TensorDataset

    latents = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch):
            images = torch.stack(
                [
                    dataset_tensor(dataset, index)
                    for index in range(start, min(len(dataset), start + batch))
                ]
            )
            latents.append(backend.encode(images.to(device)).cpu())
    return TensorDataset(torch.cat(latents, dim=0))


def train_task_classifier(
    task: TaskConfig,
    classifier: nn.Module,
    schedule: DiffusionSchedule,
    source_dataset,
    target_dataset_,
    args: ExperimentArgs,
    device: str,
    logger: Optional[Logger] = None,
) -> nn.Module:
    """Fine-tune the binary source/target classifier on noised images."""
    config = task.classifier_config()
    config.freeze_backbone = args.classifier_freeze_backbone
    labelled = LabelledImageDataset(source_dataset, target_dataset_)
    batches = infinite_loader(labelled, batch_size=config.batch_size, shuffle=True)
    train_domain_classifier(
        classifier, schedule, batches, config=config, device=device, logger=logger
    )
    return classifier


def run_task(args: Optional[ExperimentArgs] = None, logger: Optional[Logger] = None) -> Dict:
    """Run one full experiment (Algorithm 1 + evaluation)."""
    args = args or ExperimentArgs()
    task = get_task(args.task)
    device = str(get_device(args.device))
    set_seed(args.seed)
    output_dir = ensure_dir(os.path.join(args.output_dir, task.name))
    logger = logger or Logger(os.path.join(output_dir, "log.txt"))
    logger.log(f"[exp] task={task.name} backbone={task.backbone} device={device}")

    if args.target_root is None:
        raise ValueError("--target-root is required (folder with the 10 target images)")

    # ---- data --------------------------------------------------------
    targets = target_dataset(
        task.target, args.target_root, image_size=task.image_size, num_shots=task.num_shots, seed=args.seed
    )
    target_images = torch.stack([targets[index] for index in range(len(targets))])
    source_dataset = None
    if args.source_root is not None:
        from .data import ImageTensorDataset

        source_paths = list_images(args.source_root)
        source_dataset = ImageTensorDataset(source_paths, image_size=task.image_size)
        logger.log(f"[data] {len(source_dataset)} source images, {len(targets)} target images")

    # ---- backbone + adaptors ----------------------------------------
    model, schedule, in_channels, is_latent = build_backbone(task, args, device, logger)
    latent_backend = model if is_latent else None
    adaptor_target = model.unet if is_latent else model

    if is_latent:
        example_input = torch.randn(1, in_channels, task.image_size // 8, task.image_size // 8, device=device)
    else:
        example_input = torch.randn(1, in_channels, task.image_size, task.image_size, device=device)
    forward_kwargs = {"timesteps": torch.zeros(1, dtype=torch.long, device=device)}
    adaptor_config = task.ant_config().adaptor
    add_adaptors(
        adaptor_target,
        adaptor_config,
        example_input=example_input,
        forward_kwargs=forward_kwargs,
        containers=args.adaptor_containers,
        granularity=args.adaptor_granularity,
        verbose=True,
    )
    n_adaptor = sum(p.numel() for name, p in model.named_parameters() if ".adaptor." in name)
    logger.log(
        f"[adaptor] {n_adaptor/1e6:.2f}M trainable adaptor parameters "
        f"(parameter rate {100 * parameter_rate(model):.2f}%)"
    )

    # ---- classifier --------------------------------------------------
    if args.classifier_checkpoint is not None and os.path.exists(args.classifier_checkpoint):
        classifier = load_trained_domain_classifier(args.classifier_checkpoint, device=device)
    else:
        classifier = load_domain_classifier(
            checkpoint=None, download=True, num_classes=2, device=device
        )
    if source_dataset is not None:
        classifier = train_task_classifier(
            task,
            classifier,
            schedule,
            source_dataset,
            targets,
            args,
            device,
            logger=logger,
        )
    elif args.baseline != "vanilla":
        raise ValueError(
            "--source-root is required: the domain classifier of Section 5.2 is "
            "fine-tuned on source vs. target images"
        )
    if is_latent:
        # the classifier is trained on noised images (Section 5.5); for the LDM
        # it is applied to noised latents by decoding them first
        classifier = PixelSpaceClassifier(classifier, latent_backend)
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    classifier.eval()

    # ---- transfer ----------------------------------------------------
    ant_config = task.ant_config()
    ant_config.seed = args.seed
    ant_config.save_every = args.save_every
    ant_config.save_dir = os.path.join(output_dir, "checkpoints")

    if is_latent:
        train_space_dataset = encode_dataset_to_latents(latent_backend, targets, device)
    else:
        train_space_dataset = targets
    batches = infinite_loader(train_space_dataset, batch_size=ant_config.batch_size, shuffle=True)

    if args.baseline == "full":
        full_model = model.unet if is_latent else model
        history = train_full_finetune(
            full_model, classifier, schedule, batches, ant_config, device, logger, in_channels=in_channels
        )
    else:
        if args.baseline == "vanilla":
            ant_config.use_similarity_guidance = False
            ant_config.use_adversarial_noise = False
        trainer = ANTTrainer(
            model,
            classifier,
            schedule,
            ant_config,
            device=device,
            logger=logger,
            in_channels=in_channels,
        )
        history = trainer.train(batches)
        trainer.save_checkpoint()

    # ---- evaluation --------------------------------------------------
    logger.log("[eval] generating samples")
    steps = None
    if args.ddim_steps:
        from .schedules import respaced_timesteps

        steps = respaced_timesteps(schedule.num_timesteps, f"ddim{args.ddim_steps}", device=device)
        steps = torch.flip(steps, dims=[0])

    latents_shape = (
        (args.num_generated, in_channels, task.image_size // 8, task.image_size // 8)
        if is_latent
        else (args.num_generated, in_channels, task.image_size, task.image_size)
    )
    with torch.no_grad():
        samples = sample_loop(
            schedule,
            latents_shape,
            eps_predictor(model, in_channels=in_channels),
            steps=steps,
            eta=0.0 if steps is not None else 1.0,
            device=device,
        )
        if is_latent:
            images = torch.cat(
                [
                    latent_backend.decode(samples[start : start + 8])
                    for start in range(0, samples.shape[0], 8)
                ]
            )
        else:
            images = samples

    save_image_grid(images[:16], os.path.join(output_dir, "samples.png"), nrow=4)
    torch.save(images.cpu(), os.path.join(output_dir, "samples.pt"))

    results: Dict[str, object] = {
        "task": task.name,
        "backbone": task.backbone,
        "iterations": ant_config.iterations,
        "adaptor_parameters": n_adaptor,
        "final_loss": history["loss"][-1] if history.get("loss") else None,
    }
    lpips_metric = LPIPSMetric(device=device)
    results["intra_lpips"] = intra_lpips(images, target_images, metric=lpips_metric, device=device)
    logger.log(f"[eval] Intra-LPIPS = {results['intra_lpips']:.4f}")

    if args.fid_real_root is not None:
        real_paths = list_images(args.fid_real_root)
        if len(real_paths) >= args.num_fid_images:
            real_paths = real_paths[: args.num_fid_images]
        else:
            real_paths = real_paths * (args.num_fid_images // max(1, len(real_paths)) + 1)
            real_paths = real_paths[: args.num_fid_images]
        from .data import ImageTensorDataset

        real_dataset = ImageTensorDataset(real_paths, image_size=task.image_size)
        real_images = torch.stack(
            [real_dataset[index] for index in range(min(len(real_dataset), args.num_fid_images))]
        )
        fid_metric = FIDMetric(device=device)
        results["fid"] = compute_fid(images, real_images, device=device, metric=fid_metric)
        logger.log(f"[eval] FID = {results['fid']:.4f}")

    with open(os.path.join(output_dir, "results.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    return results
