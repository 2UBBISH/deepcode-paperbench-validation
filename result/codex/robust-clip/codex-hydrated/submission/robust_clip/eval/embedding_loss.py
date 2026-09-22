"""Embedding-loss evaluation (App. C.4, Table 14 -- appendix only).

Measures how far a fine-tuned encoder moves away from the original CLIP
embedding, both on clean inputs,

    L_clean(x) = || phi_FT(x) - phi_org(x) ||_2^2,

and on adversarially perturbed inputs (APGD, eps = 4/255),

    L_adv(x)   = max_{||z - x||_inf <= eps} || phi_FT(z) - phi_org(x) ||_2^2.

The appendix reports that TeCoA already distorts the *clean* embedding heavily
(E[L_clean] = 236.9 / 292.7 for eps = 2/255 / 4/255) while FARE keeps it small
(32.7 / 47.6), which is the quantitative version of Theorem 3.1.
"""
from __future__ import annotations

import argparse
import json

import torch

from ..attacks.apgd import APGDAttack
from ..models import load_clip_encoder
from ..training.data import build_imagenet_dataset
from ..models import build_pixel_transforms
from ..utils.common import LOGGER, add_common_args, get_device, parse_epsilon, set_seed


def embedding_losses(encoder, reference, loader, eps: float, n_iter: int, device) -> dict:
    clean_total = 0.0
    adv_total = 0.0
    count = 0
    for images, _ in loader:
        images = images.to(device)
        with torch.no_grad():
            ref = reference.encode_image(images).detach()
        clean = (encoder.encode_image(images) - ref).pow(2).sum(dim=1)

        def loss_fn(x_adv: torch.Tensor) -> torch.Tensor:
            # the attacker maximises the embedding distance to the *original*
            # CLIP embedding of the clean image
            return (encoder.encode_image(x_adv) - ref).pow(2).sum(dim=1)

        attack = APGDAttack(eps=eps, n_iter=n_iter, alpha=eps, momentum=0.9)
        _, adv = attack.perturb(loss_fn, images, maximize=True)
        clean_total += float(clean.sum())
        adv_total += float(adv.sum())
        count += images.shape[0]
    return {
        "L_clean": clean_total / max(1, count),
        "L_adv": adv_total / max(1, count),
        "num_samples": count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-key", default=None)
    parser.add_argument("--imagenet-root", default=None)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--eps", default="4/255")
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--output", default=None)
    add_common_args(parser)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    _, eval_tf = build_pixel_transforms(args.image_size, args.arch)
    dataset = build_imagenet_dataset(
        eval_tf, root=args.imagenet_root, split="validation", use_hf=args.imagenet_root is None
    )
    dataset = torch.utils.data.Subset(dataset, range(min(args.num_samples, len(dataset))))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    reference = load_clip_encoder(
        arch=args.arch, pretrained=args.pretrained, image_size=args.image_size, device=device
    )
    encoder = load_clip_encoder(
        arch=args.arch,
        pretrained=args.pretrained,
        image_size=args.image_size,
        checkpoint=args.checkpoint,
        checkpoint_key=args.checkpoint_key,
        device=device,
    )
    results = embedding_losses(encoder, reference, loader, parse_epsilon(args.eps), args.n_iter, device)
    results["checkpoint"] = args.checkpoint
    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
