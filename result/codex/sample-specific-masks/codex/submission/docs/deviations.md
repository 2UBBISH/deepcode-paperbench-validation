# Deviations, assumptions and open questions

This file records every place where the reproduction had to make a decision
because the markdown of the paper (and its addendum) does not fully determine
the implementation.  None of these changes the method itself.

## 1. Mask generator channel widths (Figures 8/9 unavailable)

Table 4 gives the parameter budgets (26,499 for the 5-layer CNN, 102,339 for
the 6-layer CNN) but the per-layer widths are only shown in Figures 8/9, and the
figure files are not part of the provided markdown.  Defaults:

| variant | hidden channels | parameters | Table 4 | deviation |
|---|---|---|---|---|
| 5-layer (ResNet-18/50) | (24, 32, 32, 32) | 26,979 | 26,499 | +1.8% |
| 6-layer (ViT-B/32) | (16, 64, 64, 64, 32) | 102,915 | 102,339 | +0.6% |

`hidden_channels` is exposed in the config, and
`python scripts/param_stats.py --search` lists channel configurations that hit
the tabulated counts exactly.

## 2. Last layer of the mask generator and Proposition 4.3

Appendix B.2 formalises the last layer as `W_last f''(r(x)) + b_last` with
`b_last in R^{H*W*C}`, i.e. an affine term of the *output* size; the proof sets
`W_last = 0` and `b_last = M`.  A standard convolution bias has one entry per
channel, so:

* the default architecture (plain convolution bias) exactly represents the
  masks that are constant per channel, which includes the full watermarking
  mask `M = J` used as the main shared-mask baseline (and as the
  "Only delta" ablation);
* `MaskGenerator(spatial_bias=True)` adds the paper's output-space affine term;
  with it, *any* shared binary mask (Narrow/Medium/Pad borders included) is
  representable, and the inclusion is verified exactly in
  `smm.theory.verify_shared_mask_inclusion`.

The theory verification uses the `spatial_bias=True` formalisation; the
training code uses the plain convolutional generator, which is what the
described architecture (3x3 convolutions + max pooling) implies.

## 3. `delta` is initialised to zero

Algorithm 1 sets `delta <- {0}^{d_P}`.  With `f_in = r(x) + delta * f_mask`, the
mask generator therefore receives zero gradient in the first optimisation step
and starts receiving gradient once `delta` is non-zero.  This is faithful to the
paper; it is asserted by `tests/test_reprogram.py`.

## 4. Patch-wise interpolation and gradients

The module is implemented as a value copy (`repeat_interleave`) rather than
bilinear/bicubic interpolation, matching Section 3.3 ("ensuring the same values
within each patch") and Appendix A.3 ("exclusively involves copying
operations").  A literal autograd implementation of a copy has a trivial
Jacobian; the mask generator must remain trainable, so gradients are not
detached.  Non-divisible sizes replicate the closest patches, as described in
Section 3.3.

## 5. Dataset splits

* `cifar10`, `cifar100`, `svhn`, `gtsrb`: official splits, matching Table 6
  exactly (50k/10k, 50k/10k, 73,257/26,032, 39,209/12,630).
* The remaining eight datasets use the benchmark splits of Chen et al. (2023)
  in the original paper; those split files are not part of the paper markdown
  and are not redistributed here.  This repository rebuilds class-balanced
  splits with the Table-6 sizes (deterministic, seed 0, cached in
  `data/splits/<dataset>_{train,test}.txt`), or consumes externally supplied
  split files through `--split-dir`.  The realised sizes are within a few
  samples of Table 6 and `scripts/make_splits.py` prints them for comparison.
* `gtsrb`: `torchvision` ships a reduced training split (26,640 images).
  `smm/datasets.py` parses the official archive when it is present so that the
  full 39,209-image training set is used, and falls back to `torchvision` with a
  warning otherwise.
* `ucf101`: videos are turned into images by taking the centre frame of each
  clip of the official fold-1 split.  The paper treats UCF101 as an image
  classification task (128x128).

## 6. Pre-trained ViT-B/32 at 384x384

The paper feeds ViT-B/32 with 384x384 inputs (Table 4 and the addendum
transform).  `torchvision`'s checkpoint is pre-trained at 224x224, so the
positional embedding is bicubically resized from the 14x14 to the 12x12 patch
grid (`smm/models.py:resize_position_embeddings`), the standard practice for
resolution change; `--arch timm_384` selects `timm`'s native
`vit_base_patch32_384` checkpoint when `timm` is available.

## 7. Baselines

* `Narrow` / `Medium` use border widths 28 and 56 as stated in Section 5
  ("a width of 28 (1/8 of the input image size)" and "a quarter of the size").
* `Pad` centres an image resized to `imgsize - 2 * pad_width` and trains the
  pattern on the border of width `pad_width` (default 28, configurable with
  `--pad-width`); the paper describes padding-based reprogramming as
  "centering the original image and adding the noise pattern around the
  images" without fixing the width.
* All methods share the same pattern `delta` semantics and are trained with the
  identical schedule (Section 5).

## 8. Optimiser details not specified in the paper

The paper states the learning rates, decay and milestones but not the optimiser
or momentum.  This reproduction follows Chen et al. (2023) (the reference for
the training protocol) with SGD, momentum 0.9 and no weight decay; momentum and
weight decay are configurable.

## 9. Not reproduced (out of scope per the addendum)

Figures 1, 2 and 6, the "Visualization of SMM, shared patterns and output
reprogrammed images" subsection, and the appendix-only experiments
(Table 5 interpolation benchmark, Table 11 mask generator size sweep,
Table 12 StanfordCars, Tables 13-14 finetuning comparisons).  Helper code for
the visualisations exists in `smm/visualization.py` but no figure is claimed.
