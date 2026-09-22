"""Adversarial fine-tuning of the CLIP vision encoder (FARE / TeCoA).

Reproduces Sec. 4 / App. B.1 of the paper:

* backbone ``ViT-L-14`` (OpenAI weights), resolution ``224 x 224``;
* **2 epochs** on ImageNet with 10 PGD steps of the inner maximisation;
* :math:`\\ell_\\infty` radius ``2/255`` or ``4/255``, PGD step size ``1/255``;
* AdamW with :math:`\\beta_1, \\beta_2 = 0.9, 0.95`, weight decay ``1e-4``,
  effective batch size 128, peak LR ``1e-5`` and a cosine schedule with a
  linear warm-up to the peak LR at 7% of the total number of steps.

FARE (``--method fare``) is unsupervised: only the images are used, the labels
are ignored and the target is the embedding of the *original* CLIP on the clean
image (Eq. (3)).  TeCoA (``--method tecoa``) is the supervised baseline of Mao
et al. (2023) (Eq. (2)).

Example
-------
::

    python -m robust_clip.training.train --method fare --eps 2/255 \\
        --epochs 2 --batch-size 128 --lr 1e-5 --output-dir runs/fare2_vitl14

"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

import open_clip

from ..models import CLIPImageEncoder, CLIPSpec, build_clip, build_pixel_transforms
from ..utils.common import (
    LOGGER,
    add_common_args,
    ensure_dir,
    get_device,
    human_readable_epsilon,
    parse_epsilon,
    set_seed,
)
from .data import (
    build_imagenet_dataset,
    build_train_loader,
    imagenet_class_names,
    imagenet_text_embeddings,
    load_class_names_file,
)
from .losses import FARELoss, TeCoALoss


# --------------------------------------------------------------------------- #
#                               LR schedule                                    #
# --------------------------------------------------------------------------- #
def build_scheduler(optimizer, total_steps: int, warmup_frac: float = 0.07, schedule: str = "cosine"):
    """Linear warm-up to the peak LR at ``warmup_frac`` of the run, then decay."""
    warmup_steps = max(1, int(round(warmup_frac * total_steps)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        if schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if schedule == "constant":
            return 1.0
        if schedule == "linear":
            return 1.0 - progress
        raise ValueError(f"unknown schedule '{schedule}'")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
#                                  training                                    #
# --------------------------------------------------------------------------- #
def train_one_epoch(
    encoder: CLIPImageEncoder,
    criterion,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    epochs: int,
    use_labels: bool,
    grad_accum_steps: int = 1,
    amp_dtype=None,
    max_steps: int | None = None,
    log_interval: int = 50,
    scaler=None,
) -> dict:
    encoder.train()
    running = {"loss": 0.0, "n": 0}
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            if use_labels:
                loss, x_adv = criterion(images, labels)
            else:
                loss, x_adv = criterion(images)

        scaled_loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder.trainable_parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running["loss"] += float(loss.detach())
        running["n"] += 1
        if step % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            LOGGER.info(
                "epoch %d/%d step %d/%d loss %.4f lr %.3e",
                epoch,
                epochs,
                step,
                len(loader),
                float(loss.detach()),
                lr,
            )
    if running["n"]:
        running["loss"] /= running["n"]
    return running


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=["fare", "tecoa"], default="fare")
    parser.add_argument("--arch", default="ViT-L-14", help="CLIP architecture (ViT-L-14 / ViT-B-32 / ...)")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--eps", default="2/255", help="training radius, e.g. '2/255' or '4/255'")
    parser.add_argument("--pgd-alpha", default="1/255", help="PGD step size (paper: 1/255)")
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-momentum", type=float, default=0.9)
    parser.add_argument(
        "--pgd-grad-norm", default="elementwise_sign",
        choices=["elementwise_sign", "mean_abs", "l1", "l2", "none"],
    )
    parser.add_argument(
        "--pgd-random-init", action=argparse.BooleanOptionalAction, default=True,
        help="initialise the inner maximisation with a uniform random perturbation",
    )
    parser.add_argument("--pgd-keep-best", action="store_true")
    parser.add_argument("--pgd-quantize-bits", type=int, default=None)
    parser.add_argument(
        "--feature",
        default="projected_class_token",
        choices=["projected_class_token", "class_token"],
        help="which CLIP image embedding the FARE loss is computed on (App. B.1: class token)",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup-frac", type=float, default=0.07)
    parser.add_argument("--lr-schedule", default="cosine", choices=["cosine", "linear", "constant"])
    parser.add_argument("--imagenet-root", default=None, help="local ImageNet root or HF dataset path")
    parser.add_argument("--imagenet-class-names", default=None, help="txt file with the 1000 class names")
    parser.add_argument("--imagenet-prompt", default="A photo of a {}.")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--tecoa-temperature", type=float, default=100.0,
        help="scale of the cosine logits in Eq. (2); the paper writes "
             "f_k = cos(phi(x), psi(t_k)) without a scale, CLIP's zero-shot "
             "classifier (and therefore the TeCoA baseline) uses the learned "
             "logit scale ~100.  Pass 1.0 for the literal reading.",
    )
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="debugging / smoke tests")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-every", type=int, default=1, help="save a checkpoint every N epochs")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--ddp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="distributed data parallel training (default: on when launched with torchrun)",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--config",
        default=None,
        help="JSON file with defaults (e.g. configs/fare_vitl14_eps2.json); "
        "explicit command line arguments take precedence",
    )
    add_common_args(parser)
    return parser


def merge_config(args, parser: argparse.ArgumentParser, argv=None):
    """Overlay a JSON config on top of the parsed arguments (CLI wins)."""
    if not args.config:
        return args
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    known = {action.dest for action in parser._actions}
    provided = set()
    for token in (argv if argv is not None else sys.argv[1:]):
        if token.startswith("--"):
            provided.add(token[2:].split("=")[0].replace("-", "_"))
    for key, value in config.items():
        if key in known and key not in provided:
            setattr(args, key, value)
    LOGGER.info("loaded configuration from %s", args.config)
    return args


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    merge_config(args, parser, argv)
    set_seed(args.seed)
    device = get_device(args.device)
    LOGGER.info("device: %s", device)
    ensure_dir(args.output_dir)

    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    eps = parse_epsilon(args.eps)
    alpha = parse_epsilon(args.pgd_alpha)

    # ------------------------------------------------------------ model(s) --
    spec = CLIPSpec.from_arch(args.arch, image_size=args.image_size, pretrained=args.pretrained)
    model, _, _, tokenizer = build_clip(spec)
    encoder = CLIPImageEncoder(model, feature=args.feature).to(device)
    if args.gradient_checkpointing and hasattr(encoder.visual, "set_grad_checkpointing"):
        encoder.visual.set_grad_checkpointing(True)
        LOGGER.info("enabled gradient checkpointing for the vision encoder")

    # The *original* CLIP encoder provides the fixed targets of the FARE loss.
    reference_encoder = None
    if args.method == "fare":
        ref_model, _, _, _ = build_clip(spec)
        reference_encoder = CLIPImageEncoder(ref_model, feature=args.feature).to(device)
        reference_encoder.eval()
        for param in reference_encoder.parameters():
            param.requires_grad_(False)
        LOGGER.info("FARE: fixed reference = original CLIP (phi_org), labels are not used")

    train_tf, _ = build_pixel_transforms(args.image_size, args.arch)
    dataset = build_imagenet_dataset(
        train_tf, root=args.imagenet_root, split=args.dataset_split,
    )
    if args.max_train_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_train_samples, len(dataset))))
    sampler = None
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and (args.ddp is not False):
        import torch.distributed as dist

        if dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
            )
    loader = build_train_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        sampler=sampler,
    )
    LOGGER.info("training on %d images", len(dataset))

    if args.method == "fare":
        criterion = FARELoss(
            encode_fn=encoder.encode_image,
            reference_encoder=reference_encoder,
            eps=eps,
            alpha=alpha,
            n_steps=args.pgd_steps,
            random_init=args.pgd_random_init,
            momentum=args.pgd_momentum,
            grad_normalization=args.pgd_grad_norm,
            keep_best=args.pgd_keep_best,
            quantize_bits=args.pgd_quantize_bits,
        )
        use_labels = False
    else:
        class_names = None
        if args.imagenet_class_names:
            class_names = load_class_names_file(args.imagenet_class_names)
        else:
            class_names = imagenet_class_names(dataset)
        if class_names is None:
            raise RuntimeError(
                "TeCoA needs the 1000 ImageNet class names; pass --imagenet-class-names"
            )
        LOGGER.info("encoding the fixed ImageNet text embeddings (%d classes)", len(class_names))
        text_emb = imagenet_text_embeddings(
            encoder, tokenizer, class_names, template=args.imagenet_prompt, device=device
        )
        criterion = TeCoALoss(
            encode_fn=encoder.encode_image,
            text_embeddings=text_emb,
            eps=eps,
            alpha=alpha,
            n_steps=args.pgd_steps,
            temperature=args.tecoa_temperature,
            random_init=args.pgd_random_init,
            momentum=args.pgd_momentum,
            grad_normalization=args.pgd_grad_norm,
            keep_best=args.pgd_keep_best,
            quantize_bits=args.pgd_quantize_bits,
        )
        use_labels = True

    params = encoder.trainable_parameters()

    # ------------------------------------------------------ (optional) DDP ----
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = args.ddp if args.ddp is not None else world_size > 1
    if use_ddp:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
        encoder = torch.nn.parallel.DistributedDataParallel(
            encoder, device_ids=[local_rank] if device.type == "cuda" else None
        )
        # The losses call the encoder as a module so that DDP.forward (and
        # therefore the gradient all-reduce) is used.
        criterion.encode = encoder
        LOGGER.info("DDP enabled (world size %d, local rank %d)", world_size, local_rank)

    optimizer = torch.optim.AdamW(
        params, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay
    )
    steps_per_epoch = args.max_steps_per_epoch or len(loader)
    total_steps = max(1, steps_per_epoch * args.epochs // max(1, args.grad_accum_steps))
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_frac, args.lr_schedule)

    amp_dtype = None
    scaler = None
    if args.precision == "fp16" and device.type == "cuda":
        amp_dtype = torch.float16
        # fp16 needs loss scaling for the gradients to be usable
        try:  # torch >= 2.0
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):  # pragma: no cover - older torch
            scaler = torch.cuda.amp.GradScaler()
    elif args.precision == "bf16" and device.type in ("cuda", "cpu"):
        amp_dtype = torch.bfloat16

    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        target = getattr(encoder, "module", encoder)  # unwrap DDP
        target.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        LOGGER.info("resumed from %s at epoch %d", args.resume, start_epoch)

    # ------------------------------------------------------------- training --
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        if sampler is not None:
            sampler.set_epoch(epoch)
        stats = train_one_epoch(
            encoder,
            criterion,
            loader,
            optimizer,
            scheduler,
            device,
            epoch,
            args.epochs,
            use_labels,
            grad_accum_steps=args.grad_accum_steps,
            amp_dtype=amp_dtype,
            max_steps=args.max_steps_per_epoch,
            log_interval=args.log_interval,
            scaler=scaler,
        )
        LOGGER.info("epoch %d done in %.1fs: %s", epoch, time.time() - started, stats)

        is_main = (not use_ddp) or int(os.environ.get("RANK", "0")) == 0
        if is_main and ((epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs):
            # e.g. "FARE2-CLIP_ViT-L-14" / "TeCoA4-CLIP_ViT-L-14" as in the paper
            tag = human_readable_epsilon(eps).split("/")[0]
            label = "FARE" if args.method == "fare" else "TeCoA"
            name = f"{label}{tag}-CLIP_{args.arch.replace('/', '_')}"
            checkpoint = {
                "model": encoder.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "eps": eps,
                "alpha": alpha,
                "method": args.method,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            path = os.path.join(args.output_dir, f"{name}.pt")
            torch.save(checkpoint, path)
            with open(os.path.join(args.output_dir, f"{name}.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "method": args.method,
                        "arch": args.arch,
                        "pretrained": args.pretrained,
                        "image_size": args.image_size,
                        "eps": eps,
                        "eps_readable": human_readable_epsilon(eps),
                        "pgd_steps": args.pgd_steps,
                        "pgd_alpha": alpha,
                        "epochs": args.epochs,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "batch_size": args.batch_size,
                        "feature": args.feature,
                    },
                    handle,
                    indent=2,
                )
            LOGGER.info("saved checkpoint to %s", path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
