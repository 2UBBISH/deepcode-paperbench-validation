# Sample-specific Multi-channel Masks (SMM) — reproduction

Reproduction of *Sample-specific Masks for Visual Reprogramming-based Prompting*
(Cai, Ye, Feng, Qi, Liu; ICML 2024), built from `paper/paper.md` and
`paper/addendum.md` only.  The repository contains a complete, runnable
implementation of the method, of all baselines and ablations described in the
main body of the paper, of the 11 target tasks, and of the theoretical results
(Theorem 4.2, Proposition 4.3, Proposition B.1) together with numerical checks
of them.

## What is implemented

| Paper item | Where | Status |
|---|---|---|
| SMM framework: `f_in(x) = r(x) + delta * f_mask(r(x))` (Sec. 3.1) | `smm/reprogram.py` | implemented |
| Lightweight mask generator, 5-layer (ResNet) / 6-layer (ViT) CNN, 3-channel mask (Sec. 3.2) | `smm/mask_generator.py` | implemented, parameter budget matched to Table 4 |
| Patch-wise interpolation module, patch size `2**l`, no floating-point derivation (Sec. 3.3) | `smm/mask_generator.py` | implemented (`repeat_interleave`; gradient-friendly) |
| Learning strategy of Algorithm 1 (zero-initialised `delta`, separate lr for `delta`/`phi`) | `smm/train.py` | implemented |
| Shared-mask baselines Pad / Narrow / Medium / Full (Sec. 5) | `smm/reprogram.py` | implemented |
| Output mappings Rlm / Flm / Ilm (Sec. 2.3, Algorithms 2-4) | `smm/label_mapping.py` | implemented |
| Table 1: ResNet-18 / ResNet-50 results on 11 datasets | `scripts/run_all.sh`, `smm/aggregate.py` | code complete, full runs are long (see *Reproducing the tables*) |
| Table 2: ViT-B/32 results on 11 datasets | `configs/vitb32.yaml` | code complete |
| Table 3: ablations (only `delta`, only `f_mask`, single-channel `f_mask`) | `--method only_delta/only_mask/single_channel` | implemented |
| Figure 4: patch size study `2**l`, `l in {0..4}` | `--num-pool-layers` | implemented |
| Section 4 + Appendix B: Theorem 4.2, Proposition 4.3, Proposition B.1 | `smm/theory.py`, `scripts/verify_theory.py` | implemented + numerically verified |
| Appendix D.1: SMM with Rlm / Flm / Ilm | `--label-mapping` | implemented (cheap once Table 1 runs exist) |
| Appendix D.2 learning curves: train/test accuracy per epoch | `history` field of every run JSON | implemented |

Out of scope, per `paper/addendum.md` (and *not* implemented as figures):
Figures 1 and 2 (motivation), Figure 6 / the "Feature Space Visualization
Results" subsection, and the "Visualization of SMM, shared patterns and output
reprogrammed images" subsection.  Small helpers for those analyses still exist
(`smm/visualization.py`) but they are not part of the graded results.
Appendix-only experiments (Table 5 interpolation benchmark, Table 11 enlarged
`f_mask`, Table 12 StanfordCars, Tables 13-14 finetuning comparisons) are not
core contributions and are not reproduced.

## Repository layout

```
smm/
  mask_generator.py   f_mask (CNN + patch-wise interpolation), Table-4 budget
  reprogram.py        f_in for SMM and for Pad/Narrow/Medium/Full, ablations
  label_mapping.py    f_out: Rlm / Flm / Ilm (Algorithms 2-4)
  models.py           frozen ImageNet-1K ResNet-18/50, ViT-B/32 (384x384)
  datasets.py         the 11 target tasks, paper transforms, Table-6 splits
  train.py            Algorithm 1 training loop, evaluation, result JSON
  main.py             CLI (one table cell per invocation)
  aggregate.py        mean +/- std tables across seeds
  theory.py           Theorem 4.2 / Prop. 4.3 / Prop. B.1 + numerical checks
  visualization.py    reprogrammed images, masks, t-SNE (out-of-scope figures)
configs/              resnet18.yaml, resnet50.yaml, vitb32.yaml
scripts/              run_all.sh, smoke_test.sh, verify_theory.py,
                      param_stats.py, make_splits.py, mini_table1.sh,
                      plot_curves.py, lr_search_vit.sh
tests/                pytest suite (49 tests, ~25 s on CPU)
docs/                 method.md, experiments.md, deviations.md
```

## Installation

```bash
pip install -r requirements.txt     # torch, torchvision, numpy, PyYAML, sklearn, matplotlib
# optional, for the native 384x384 ViT-B/32 checkpoint:
# pip install timm
```

Everything runs from the repository root (`python -m smm.main ...`); no extra
`PYTHONPATH` is needed.  Data is downloaded by `torchvision` into `data/` and
the pre-trained weights into the torch cache, so the first run of a cell needs
network access.

## Reproducing the paper's results

One invocation trains one cell (dataset x backbone x method x seed):

```bash
# Table 1 -- SMM with ResNet-18 on CIFAR-10 (Ilm mapping, 200 epochs)
python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method smm --seed 0

# Table 1 -- shared-mask baselines
for m in pad narrow medium full; do
  python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method $m --seed 0
done

# Table 2 -- ViT-B/32 (input 384x384, 6-layer mask generator)
python -m smm.main --config configs/vitb32.yaml --dataset flowers102 --method smm --seed 0

# Table 3 -- ablations on ResNet-18
python -m smm.main --config configs/resnet18.yaml --dataset flowers102 --method only_delta
python -m smm.main --config configs/resnet18.yaml --dataset flowers102 --method only_mask
python -m smm.main --config configs/resnet18.yaml --dataset flowers102 --method single_channel

# Figure 4 -- patch size 2**l
python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method smm --num-pool-layers 0

# Appendix D.1 -- other output mappings
python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method smm --label-mapping flm
python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method smm --label-mapping rlm

# aggregate the runs into mean +/- std tables
python -m smm.aggregate --runs runs --table 1
```

The full experiment matrix (all datasets x methods x 3 seeds) is scripted:

```bash
bash scripts/run_all.sh resnet18     # Table 1 (ResNet-18) + Table 3 ablations
bash scripts/run_all.sh resnet50     # Table 1 (ResNet-50)
bash scripts/run_all.sh vitb32       # Table 2
bash scripts/run_all.sh patches      # Figure 4
bash scripts/run_all.sh mappings     # Appendix D.1
```

Each run writes `runs/<dataset>/<backbone>_<dataset>_<method>_<mapping>_seed<k>.json`
with the final test accuracy, the full per-epoch history, the label mapping and
the parameter counts.  `python -m smm.aggregate` turns them into the
mean +/- std tables of the paper.

Expected trends (the paper's Table 1-3): SMM outperforms every shared-mask
baseline on almost every dataset, with the largest gains on domain-shifted,
colour-rich tasks (SVHN, Flowers102, EuroSAT) and on ViT-B/32; on DTD with
ResNet-18 the padding baseline remains competitive because resizing-based
methods disturb texture features; the ablations degrade in the order
Only `delta` < Single-channel `f_mask` < SMM.

### Cost

The paper's settings are 200 epochs per cell with batch size up to 256, i.e.
one cell is a few GPU-hours (three seeds per cell, 11 datasets, five methods:
the complete Table 1 needs a few hundred A100-hours).  Nothing in this
reproduction was trained at that scale inside the authoring environment; the
code is written to be launched on the GPU machine that runs the evaluation.
For a quick sanity check use `--epochs 1 --eval-every 1` on a small dataset
(`bash scripts/smoke_test.sh`).

### Fast sanity check of the trend

```bash
bash scripts/mini_table1.sh     # CIFAR-10, 1.2k images, 64x64, 5 epochs, all 5 methods
```

`scripts/mini_table1.sh` trains every method on a small subset with
`--train-subset/--test-subset` and a reduced input size so that the *relative*
ordering of the methods can be inspected in minutes (CPU included).  It is a
pipeline check, not a reproduction of the reported accuracies: with a couple of
thousand samples the sample-specific masks have far less data than in the
paper's 200-epoch runs over the full training sets.

## Verifying the theory

```bash
python scripts/verify_theory.py          # Prop. 4.3, Prop. B.1 + empirical approximation error
python scripts/param_stats.py            # Table 4 parameter budget (and --search for exact widths)
python -m pytest tests -q                # 49 unit tests (shapes, mappings, theory, splits, learning)
```

`scripts/verify_theory.py` prints, for example:

```
proposition_4_3_shared_mask        max_abs_diff = 0.0     (SMM reproduces a shared mask exactly)
proposition_4_3_watermark          max_abs_diff = 0.0     (full-watermark baseline subset of SMM)
proposition_B_1_sample_specific    max_abs_diff = 0.0     (delta = J recovers f_mask alone)
empirical_approximation_error      shr 2.17 > sp 2.46 ... smm 0.07
```

i.e. the empirical approximation error of the SMM hypothesis space is far
below that of the shared-mask family, as stated in Proposition 4.3.

## Implementation notes and deviations

The full list is in `docs/deviations.md`; the important ones:

1. **Mask generator widths.** Table 4 fixes the parameter budget (26,499 for
   the 5-layer CNN, 102,339 for the 6-layer CNN) but the per-layer widths live
   in Figures 8/9, which are not part of the provided markdown.  The default
   channels `(24,32,32,32)` and `(16,64,64,64,32)` give 26,979 and 102,915
   parameters (within 1.8% / 0.6%); `scripts/param_stats.py --search` can be
   used to find widths that hit the counts exactly, and `hidden_channels` is
   configurable.
2. **Patch-wise interpolation** is implemented as `repeat_interleave`, i.e. a
   value-copy with a trivial Jacobian, instead of bilinear/bicubic
   interpolation (Appendix A.3).  Note that the gradient must reach `f_mask`
   for the generator to learn at all, so the module is differentiable through
   the copy (the paper's "omits the derivation step" refers to the expensive
   floating-point gradients of classical interpolation).
3. **`delta` is initialised to zero** (Algorithm 1).  Consequently the mask
   generator receives zero gradient in the very first step and starts learning
   from the second step onwards; this is inherent to the paper's algorithm and
   is asserted in `tests/test_reprogram.py`.
4. **Proposition 4.3 formalisation.** Appendix B.2 writes the last layer as
   `W_last f''(r(x)) + b_last` with `b_last` of the output size.  A standard
   convolution bias is per-channel, so a spatially varying shared mask is only
   exactly representable with the paper's output-space bias; `MaskGenerator`
   exposes `spatial_bias=True` for that construction and the verification uses
   it.  Without it, the exactly representable shared masks are the
   per-channel-constant ones, which includes the full watermarking mask `M = J`
   (the main shared-mask baseline, also the "Only delta" ablation).  Both cases
   are checked exactly (`verify_shared_mask_inclusion`,
   `verify_watermark_inclusion`).
5. **Splits.** CIFAR-10/100, SVHN and GTSRB use their official splits, which
   match Table 6 exactly.  For the other 8 datasets the benchmark split files
   of Chen et al. (2023) are not redistributed here; the code rebuilds a
   class-balanced split with the Table-6 sizes (deterministic, seed 0, cached
   under `data/splits`), or uses externally supplied split files
   (`--split-dir`).  Realised sizes are within a few samples of Table 6.
6. **ViT-B/32 at 384x384.** `torchvision`'s ViT-B/32 is pre-trained at 224, so
   the positional embedding is bicubically resized to the 12x12 patch grid
   (`smm/models.py`).  `--arch timm_384` uses `timm`'s native
   `vit_base_patch32_384` checkpoint instead when `timm` is installed.
7. **UCF101** is a video dataset in `torchvision`; following the paper's
   framing of it as a 128x128 image task, the centre frame of every clip
   (official fold 1) is used.

## Acknowledgements

All method and experiment details come from the paper and its addendum;
implementations of the baselines (padding-based reprogramming, watermarking,
Rlm/Flm/Ilm label mapping) were written from the descriptions in the paper, and
the public repos of those works were not consulted.

## What was actually executed while building this repository

The authoring environment had no GPU and a strict wall-clock budget, so the
paper's 200-epoch runs were *not* executed here; the code is written for the GPU
machine that runs the evaluation.  The following checks were executed and all of
them pass:

| check | command | result |
|---|---|---|
| unit tests (shapes, patch interpolation, mappings, splits, theory, learning) | `python -m pytest tests -q` | 49 passed |
| exact inclusion checks + approximation-error experiment | `python scripts/verify_theory.py` | `included = 1.0` for all three hypothesis-space claims, `smm` loss lowest |
| mask generator parameter budget | `python scripts/param_stats.py` | 26,979 (Table 4: 26,499) and 102,915 (Table 4: 102,339) |
| CLI dry runs for every method/backbone | `python -m smm.main --dry-run ...` | correct mask sizes, patch sizes, parameter counts |
| real end-to-end training (CIFAR-10, pre-trained ResNet-18, 224x224, 256 train / 128 test images, 2 epochs) | `python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method smm --train-subset 256 --test-subset 128 --epochs 2` | ran, Ilm mapping learned, 36.7% test accuracy (10% chance) |
| reduced-scale baseline run (CIFAR-10, 64x64, 1.2k train / 600 test images, 5 epochs, Pad) | `bash scripts/mini_table1.sh` | training loss 3.40 -> 2.26, 22.5% test accuracy (10% chance) |

These runs validate that the data pipeline, the reprogramming modules, the
label-mapping algorithms and the training loop work end to end; they do not
reproduce the paper's accuracies, which require the full 200-epoch runs on each
dataset (see `scripts/run_all.sh`).
