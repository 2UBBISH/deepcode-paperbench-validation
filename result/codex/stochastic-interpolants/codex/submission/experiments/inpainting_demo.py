"""A small CPU demonstration of the in-painting pipeline of Section 4.1.

ImageNet itself is far too large to train on in this environment, so this
script repeats the *exact* training/sampling procedure of Section 4.1 on a
small procedural image dataset and compares

    * the data-dependent coupling   x_0 = xi o x_1 + (1 - xi) o zeta   ("Ours")
    * the independent coupling      x_0 ~ N(0, Id)                     (baseline)

with the same U-Net architecture, optimiser, mask distribution (64 tiles,
p = 0.3) and number of gradient steps.  The comparison mirrors Table 2: the
uncoupled interpolant has to reconstruct the whole image from pure noise,
whereas the coupled one always sees the observed pixels.

Reported quantities (all on held-out images):
    masked-region MSE / PSNR   (the region the model has to fill in)
    observed-region error      (must be ~0 for the coupled model)
    sample grid                (base | model sample | ground truth)

Usage:  python experiments/inpainting_demo.py --steps 600 --out results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from si_couplings.data.imagenet import build_dataloader  # noqa: E402
from si_couplings.sample import sample_inpainting  # noqa: E402
from si_couplings.train import build_experiment  # noqa: E402
from si_couplings.utils import EMA, JsonlLogger, set_seed  # noqa: E402
from si_couplings.visualize import inpainting_panel  # noqa: E402


# ---------------------------------------------------------------------------
class BlobImages(Dataset):
    """Procedural 32x32 (or 64x64) RGB images made of a few coloured blobs."""

    def __init__(self, n: int = 4096, resolution: int = 32, num_classes: int = 4, seed: int = 0):
        self.n, self.resolution, self.num_classes, self.seed = n, resolution, num_classes, seed
        cols = torch.tensor(
            [[1.0, 0.2, 0.2], [0.2, 1.0, 0.3], [0.3, 0.4, 1.0], [1.0, 0.9, 0.2]]
        )
        rows = torch.tensor(
            [[0.1, 0.9, 0.4], [0.8, 0.2, 0.7], [0.4, 0.6, 0.9], [0.9, 0.5, 0.1]]
        )
        self.palette = torch.stack([cols, rows], dim=1)  # (num_classes, 2, 3)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed * 100003 + idx)
        label = int(idx % self.num_classes)
        s = self.resolution
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s), indexing="ij")
        bg, fg = self.palette[label]
        img = bg[:, None, None] * torch.ones(3, s, s)
        for _ in range(3):
            cy, cx = torch.rand(2, generator=g) * 1.6 - 0.8
            r = 0.15 + 0.35 * float(torch.rand(1, generator=g))
            mask = torch.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r**2)))
            img = img + fg[:, None, None] * mask[None]
        img = img + 0.05 * torch.randn(3, s, s, generator=g)
        return img.clamp(0, 1) * 2 - 1, label


def config(task: str, coupling: str, resolution: int = 32) -> dict:
    if coupling == "independent" and task == "super_resolution":
        coupling = "super_resolution_baseline"
    return {
        "task": task,
        "seed": 0,
        "interpolant": "linear" if coupling in ("independent", "super_resolution_baseline")
        else "linear_zero_gamma",
        "coupling": {
            "name": coupling,
            "n_tiles": 8,
            "p_missing": 0.3,
            "channels": 3,
            "base_scale": 1.0,
            "scale": 2,
            "sigma": 0.5,
        },
        "data": {"kind": "synthetic", "image_size": resolution, "channels": 3, "num_classes": 4},
        "model": {
            "kind": "unet",
            "dim": 32,
            "dim_mults": [1, 2],
            "resnet_block_groups": 8,
            "learned_sinusoidal_cond": True,
            "learned_sinusoidal_dim": 16,
            "attn_dim_head": 16,
            "attn_heads": 2,
            "attn_resolutions": [8, 16],
            "image_size": resolution,
            "num_classes": 4,
            "data_channels": 3,
        },
        "optim": {"batch_size": 16, "lr": 5e-4, "max_steps": 600, "log_every": 100, "save_every": 600},
        "sampling": {"method": "dopri5", "steps": 50, "guidance_scale": 1.0},
    }


def train_model(cfg: dict, dataset: Dataset, steps: int, device, logger=None):
    parts = build_experiment(cfg, device=device)
    model, coupling, schedule = parts["model"], parts["coupling"], parts["schedule"]
    loader = DataLoader(dataset, batch_size=cfg["optim"]["batch_size"], shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["optim"]["lr"])
    ema = EMA(model, beta=0.995)
    step = 0
    model.train()
    while step < steps:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch = coupling.sample(images, labels=labels)
            from si_couplings.losses import velocity_loss

            loss, _ = velocity_loss(model, schedule, batch, mse_form=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10_000.0)
            opt.step()
            ema.update(model)
            step += 1
            if logger is not None and step % cfg["optim"]["log_every"] == 0:
                logger.log(**{f"{cfg['_tag']}_step": step, f"{cfg['_tag']}_loss": float(loss)})
            if step >= steps:
                break
    ema.model.eval()
    return ema.model, coupling


@torch.no_grad()
def evaluate(model, coupling, images: Tensor, labels: Tensor, *, task: str, steps: int = 50,
             xi: Optional[Tensor] = None) -> dict:
    if task == "super_resolution":
        from si_couplings.data.superres import downsample
        from si_couplings.sample import sample_super_resolution

        low = downsample(images, coupling.scale)
        out = sample_super_resolution(model, coupling, low, labels=labels, steps=steps,
                                      high_res_truth=images)
        sample = out["sample"].clamp(-1, 1)
        mse = F.mse_loss(sample, images).item()
        return {"mse": mse, "psnr": 10 * torch.log10(torch.tensor(4.0 / max(mse, 1e-8))).item(),
                "result": out}

    out = sample_inpainting(model, coupling, images, xi=xi, labels=labels, steps=steps)
    sample = out["sample"].clamp(-1, 1)
    mask = out["mask"]  # 1 where the model must fill in
    xi = out["xi"]
    denom = mask.sum().clamp(min=1.0)
    filled_mse = (((sample - images) ** 2) * mask).sum().item() / denom.item()
    observed_err = (((sample - images).abs()) * xi).max().item()
    # variance of the ground truth inside the missing region (a trivial model
    # that outputs the conditional mean would reach this MSE)
    return {
        "masked_mse": filled_mse,
        "masked_psnr": 10 * torch.log10(torch.tensor(4.0 / max(filled_mse, 1e-8))).item(),
        "observed_max_error": observed_err,
        "result": out,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="inpainting", choices=["inpainting", "super_resolution"])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-test", type=int, default=32)
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)

    set_seed(0)
    device = torch.device("cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir / "inpainting_demo.jsonl")

    train_ds = BlobImages(args.n_train, args.resolution)
    test_ds = BlobImages(args.n_test, args.resolution, seed=7)
    test_images = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    test_labels = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
    test_xi = None
    if args.task == "inpainting":
        from si_couplings.couplings import tile_mask

        torch.manual_seed(1234)
        test_xi = tile_mask(
            len(test_ds), 3, args.resolution, args.resolution, n_tiles=8, p_missing=0.3
        )

    results = {}
    panels = []
    for coupling_name, tag in [("independent", "baseline"), ("inpainting" if args.task == "inpainting"
                                                               else "super_resolution", "ours")]:
        cfg = config(args.task, coupling_name, args.resolution)
        cfg["_tag"] = tag
        model, coupling = train_model(cfg, train_ds, args.steps, device, logger)
        metrics = evaluate(model, coupling, test_images, test_labels, task=args.task,
                           steps=cfg["sampling"]["steps"], xi=test_xi)
        panel = metrics.pop("result")
        panels.append((tag, panel))
        results[tag] = metrics
        print(tag, {k: round(v, 4) for k, v in metrics.items()})

    (out_dir / f"{args.task}_demo.json").write_text(json.dumps(results, indent=2))
    for tag, panel in panels:
        # keep the raw tensors so that the panels can be re-drawn cheaply
        torch.save(panel, out_dir / f"{args.task}_demo_{tag}.pt")
        path = out_dir / f"{args.task}_demo_{tag}.png"
        if args.task == "inpainting":
            inpainting_panel(panel, path, max_items=4)
        else:
            from si_couplings.visualize import super_resolution_panel

            super_resolution_panel(panel, path, max_items=4)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
