"""Small-scale verification of the FARE mechanism (a few minutes on CPU).

It fine-tunes a pretrained CLIP ViT-B/32 with a handful of FARE steps on a small
subset of one dataset and reports

* the clean embedding loss ``E[L_clean]`` and the adversarial embedding loss
  ``E[L_adv]`` (App. C.4 / Table 14) before and after fine-tuning, and
* the clean and adversarial zero-shot accuracy before and after fine-tuning.

The point is *not* to reproduce the numbers of the paper (that needs the full
ImageNet fine-tuning of App. B.1) but to show that the loss does what Theorem 3.1
and Table 14 say: it keeps the clean embedding close to the original CLIP while
reducing the embedding distortion on adversarial inputs.

Example::

    python scripts/quick_fare_check.py --steps 6 --samples 16 --eval-samples 60 \
        --batch-size 8 --pgd-steps 5 --attack-iter 5 --lr 1e-6 --device cpu

Reference output of that command (CPU, ~4 min)::

    E[L_clean]        0.00  ->        2.67
    E[L_adv]        141.34  ->      129.98
    clean acc         88.3% ->        85.0%
    robust acc        21.7% ->        25.0%

i.e. exactly the qualitative behaviour of Table 14 (the clean embedding barely
moves while the adversarial embedding loss shrinks) and Table 4 (robustness up,
clean accuracy almost unchanged) -- on a scale of 16 images and 6 steps.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robust_clip.attacks.apgd import APGDAttack
from robust_clip.models import build_pixel_transforms, load_clip_encoder
from robust_clip.training.losses import FARELoss, TeCoALoss
from robust_clip.utils.common import LOGGER, get_device, parse_epsilon, set_seed


def load_images(dataset: str, num_samples: int, resolution: int):
    from datasets import load_dataset

    if dataset != "cifar10":
        raise ValueError(f"unsupported dataset '{dataset}'")
    raw = load_dataset("cifar10", split="test")
    subset = raw.select(range(num_samples))
    _, transform = build_pixel_transforms(resolution, "ViT-B-32")
    images = torch.stack([transform(image) for image in subset["img"]])
    labels = torch.tensor(subset["label"])
    return images, labels, raw.features["label"].names


def zero_shot_accuracy(encoder, tokenizer, images, labels, class_names, eps, n_iter, attack: bool):
    prompts = [f"a photo of a {name}." for name in class_names]
    with torch.no_grad():
        text = encoder.encode_text(tokenizer(prompts))
        text = text / text.norm(dim=-1, keepdim=True)

    def logits_fn(x):
        return F.normalize(encoder.encode_image(x), dim=-1) @ text.t()

    with torch.no_grad():
        if not attack:
            return 100.0 * float((logits_fn(images).argmax(dim=1) == labels).float().mean())
    if not attack:
        return 0.0
    # the attack needs gradients w.r.t. the input
    attack_obj = APGDAttack(eps=eps, n_iter=n_iter, alpha=eps)
    x_adv, _ = attack_obj.perturb(
        lambda x: F.cross_entropy(logits_fn(x), labels, reduction="none"),
        images,
        maximize=True,
    )
    with torch.no_grad():
        return 100.0 * float((logits_fn(x_adv).argmax(dim=1) == labels).float().mean())


def embedding_losses(encoder, reference, images, eps, n_iter):
    with torch.no_grad():
        ref = reference.encode_image(images)
        clean = float((encoder.encode_image(images) - ref).pow(2).sum(dim=1).mean())

    def objective(x_adv):
        return (encoder.encode_image(x_adv) - ref).pow(2).sum(dim=1)

    attack = APGDAttack(eps=eps, n_iter=n_iter, alpha=eps)
    _, adv = attack.perturb(objective, images, maximize=True)
    return clean, float(adv.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument(
        "--method", default="fare", choices=["fare", "tecoa"],
        help="TeCoA is the supervised baseline of Mao et al. (2023)",
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--eps", default="4/255")
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument(
        "--lr", type=float, default=1e-6,
        help="the paper uses 1e-5 for the full 2-epoch ImageNet run; with only a "
             "handful of images the gradient is dominated by noise, so Adam's "
             "first (sign-like) steps hurt the model unless the LR is reduced",
    )
    parser.add_argument("--attack-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    eps = parse_epsilon(args.eps)
    import open_clip

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    encoder = load_clip_encoder("ViT-B-32", device=device)
    reference = load_clip_encoder("ViT-B-32", device=device)
    for param in reference.parameters():
        param.requires_grad_(False)
    reference.eval()

    train_images, train_labels, class_names = load_images(args.dataset, args.samples, args.resolution)
    eval_images, eval_labels, _ = load_images(args.dataset, args.eval_samples, args.resolution)
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    eval_images = eval_images.to(device)
    eval_labels = eval_labels.to(device)

    LOGGER.info("measuring the embedding losses and accuracies before fine-tuning")
    clean_before, adv_before = embedding_losses(encoder, reference, eval_images, eps, args.attack_iter)
    acc_clean_before = zero_shot_accuracy(
        encoder, tokenizer, eval_images, eval_labels, class_names, eps, args.attack_iter, attack=False
    )
    acc_adv_before = zero_shot_accuracy(
        encoder, tokenizer, eval_images, eval_labels, class_names, eps, args.attack_iter, attack=True
    )

    if args.method == "fare":
        criterion = FARELoss(
            encode_fn=encoder,
            reference_encoder=reference,
            eps=eps,
            alpha=1 / 255,
            n_steps=args.pgd_steps,
            momentum=0.9,
            grad_normalization="elementwise_sign",
        )
    else:
        # TeCoA: supervised adversarial training on the fixed text embeddings of
        # the classes (Eq. (2)); it never sees the original CLIP embedding.
        with torch.no_grad():
            text_embeddings = encoder.encode_text(
                tokenizer([f"a photo of a {name}." for name in class_names])
            )
        criterion = TeCoALoss(
            encode_fn=encoder,
            text_embeddings=text_embeddings,
            eps=eps,
            alpha=1 / 255,
            n_steps=args.pgd_steps,
            momentum=0.9,
            grad_normalization="elementwise_sign",
        )
    optimizer = torch.optim.AdamW(
        encoder.trainable_parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4
    )
    encoder.train()
    for step in range(args.steps):
        for start in range(0, len(train_images), args.batch_size):
            batch = train_images[start:start + args.batch_size]
            if args.method == "fare":
                loss, _ = criterion(batch)
            else:
                labels = train_labels[start:start + args.batch_size]
                loss, _ = criterion(batch, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        LOGGER.info("%s step %d/%d: loss %.2f", args.method.upper(), step + 1, args.steps, float(loss))
    encoder.eval()

    LOGGER.info("measuring the embedding losses and accuracies after fine-tuning")
    clean_after, adv_after = embedding_losses(encoder, reference, eval_images, eps, args.attack_iter)
    acc_clean_after = zero_shot_accuracy(
        encoder, tokenizer, eval_images, eval_labels, class_names, eps, args.attack_iter, attack=False
    )
    acc_adv_after = zero_shot_accuracy(
        encoder, tokenizer, eval_images, eval_labels, class_names, eps, args.attack_iter, attack=True
    )

    print()
    print(
        f"method={args.method} dataset={args.dataset} samples={args.samples} "
        f"steps={args.steps} eps={args.eps} lr={args.lr:g}"
    )
    print(f"E[L_clean]  {clean_before:10.2f}  ->  {clean_after:10.2f}   (should stay small)")
    print(f"E[L_adv]    {adv_before:10.2f}  ->  {adv_after:10.2f}   (should decrease)")
    print(f"clean acc   {acc_clean_before:10.1f}% ->  {acc_clean_after:10.1f}%")
    print(f"robust acc  {acc_adv_before:10.1f}% ->  {acc_adv_after:10.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
