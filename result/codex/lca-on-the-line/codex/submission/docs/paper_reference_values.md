# Reference values from the paper

Numbers transcribed from the ICML 2024 paper so that a reproduction run can be
compared against them at a glance.  Trends, not exact digits, are what the
evaluation looks at.

## Table 1 — mistake severity (selected models)

| Model | ImgN LCA ↓ | ImgN Top1 ↑ | ImgN-v2 Top1 | ImgN-S Top1 | ImgN-R Top1 | ImgN-A Top1 | ObjNet Top1 |
|---|---|---|---|---|---|---|---|
| ResNet18 | 6.643 | 0.698 | 0.573 | 0.202 | 0.330 | 0.011 | 0.272 |
| ResNet50 | 6.539 | 0.733 | 0.610 | 0.235 | 0.361 | 0.018 | 0.316 |
| CLIP_RN50 | 6.327 | 0.579 | 0.511 | 0.332 | 0.562 | 0.218 | 0.398 |
| CLIP_RN50x4 | 6.166 | 0.641 | 0.573 | 0.415 | 0.681 | 0.384 | 0.504 |

## Table 2 — R² / PEA, 75 models

| ID | OOD | ImgN-v2 | ImgN-S | ImgN-R | ImgN-A | ObjNet |
|---|---|---|---|---|---|---|
| Top1 | Top1 | **0.962** / 0.980 | 0.075 / 0.275 | 0.020 / 0.140 | 0.009 / 0.094 | 0.273 / 0.522 |
| LCA | Top1 | 0.339 / 0.582 | **0.816** / 0.903 | **0.779** / 0.883 | **0.704** / 0.839 | **0.915** / 0.956 |
| Top1 | Top5 | 0.889 / 0.943 | 0.052 / 0.229 | 0.004 / 0.060 | 0.013 / 0.115 | 0.262 / 0.512 |
| LCA | Top5 | 0.445 / 0.667 | 0.811 / 0.901 | 0.738 / 0.859 | 0.799 / 0.894 | 0.924 / 0.961 |

Key claim: ID LCA keeps a strong correlation on the four severely shifted OOD
datasets (R² > 0.7, PEA > 0.83), while ID Top-1 collapses there.

## Table 3 — error-prediction MAE ↓ (75 models)

| Method | ImgN-v2 | ImgN-S | ImgN-R | ImgN-A | ObjNet |
|---|---|---|---|---|---|
| ID Top1 (Miller et al.) | **0.040** | 0.230 | 0.277 | 0.192 | 0.178 |
| AC (Hendrycks & Gimpel) | 0.043 | 0.124 | **0.113** | 0.324 | 0.127 |
| Aline-D (Baek et al.) | 0.121 | 0.270 | 0.167 | 0.409 | 0.265 |
| Aline-S (Baek et al.) | 0.072 | 0.143 | 0.201 | 0.165 | 0.131 |
| (Ours) ID LCA | 0.162 | **0.093** | 0.114 | **0.103** | **0.048** |

## Table 4 — PEA across 75 latent hierarchies

| Element | ImgN-v2 | ImgN-S | ImgN-R | ImgN-A | ObjNet |
|---|---|---|---|---|---|
| Baseline (Top1 → Top1) | **0.980** | 0.275 | 0.140 | 0.094 | 0.522 |
| WordNet (LCA → Top1) | 0.582 | **0.903** | **0.883** | **0.839** | **0.956** |
| Latent hierarchies: mean | 0.815 | 0.773 | 0.712 | 0.662 | 0.930 |
| min | 0.721 | 0.715 | 0.646 | 0.577 | 0.890 |
| max | 0.863 | 0.829 | 0.780 | 0.717 | 0.952 |
| std | 0.028 | 0.022 | 0.027 | 0.025 | 0.010 |

Key claim: latent K-means hierarchies track OOD performance almost as well as
WordNet, i.e. LCA is robust to the choice of taxonomy.

## Table 5 — WordNet soft labels for linear probing (ImageNet Top-1 / OOD Top-1)

| Backbone | ImgNet | ImgNet-V2 | ImgNet-S | ImgNet-R | ImgNet-A | ObjectNet |
|---|---|---|---|---|---|---|
| ResNet 18 baseline → ours | 69.4 → 69.4 | 56.4 → 56.9 | 19.7 → 20.7 | 31.9 → 33.8 | 1.1 → 1.2 | 27.0 → 28.0 |
| ResNet 50 | 79.5 → 79.8 | 67.9 → 68.6 | 25.5 → 27.7 | 36.5 → 42.5 | 10.3 → 16.2 | 43.2 → 45.5 |
| ViT-B | 75.8 → 75.9 | 62.9 → 62.8 | 27.0 → 27.6 | 40.5 → 41.5 | 8.0 → 8.6 | 27.6 → 28.1 |
| ViT-L | 76.8 → 76.8 | 63.9 → 63.8 | 28.4 → 29.2 | 42.2 → 43.6 | 10.6 → 11.5 | 28.7 → 29.0 |
| ConvNeXt | 82.0 → 82.1 | 70.6 → 71.0 | 28.7 → 30.0 | 42.4 → 44.3 | 21.8 → 25.3 | 44.4 → 45.5 |
| Swin | 83.1 → 83.2 | 72.0 → 71.9 | 30.3 → 31.4 | 43.5 → 45.3 | 29.5 → 32.7 | 48.3 → 49.5 |

Key claim: soft labels raise OOD accuracy without hurting ID accuracy.

## Table 6 — latent-hierarchy soft labels (ResNet-18)

| Hierarchy source | ImgNet-S | ImgNet-R | ImgNet-A | ObjectNet |
|---|---|---|---|---|
| MnasNet | 19.7 → 20.2 | 31.9 → 32.4 | 1.1 → 1.7 | 27.0 → 28.1 |
| ResNet 18 | 19.7 → 20.2 | 31.9 → 32.4 | 1.1 → 1.8 | 27.0 → 28.2 |
| vit-l-14 | 19.7 → 20.8 | 31.9 → 33.2 | 1.1 → 2.0 | 27.0 → 28.3 |
| OpenCLIP(vit-l-14) | 19.7 → 20.9 | 31.9 → 33.7 | 1.1 → 2.1 | 27.0 → 28.5 |
| WordNet | 19.7 → **21.2** | 31.9 → **35.1** | 1.1 → **1.4** | 27.0 → **28.6** |

## Table 14 — taxonomy prompts, CLIP-ViT32 (Top-1 / test CE)

| Prompt | ImageNet | ImageNet-v2 | ImageNet-S | ImageNet-R | ImageNet-A | ObjectNet |
|---|---|---|---|---|---|---|
| Baseline | 0.589 / 9.322 | 0.517 / 9.384 | 0.379 / 9.378 | 0.667 / 8.790 | 0.294 / 9.358 | 0.394 / 8.576 |
| Stack Parent | 0.381 / 9.389 | 0.347 / 9.395 | 0.219 / 9.561 | 0.438 / 9.258 | 0.223 / 9.364 | 0.148 / 9.076 |
| Shuffle Parent | 0.483 / 9.679 | 0.432 / 9.696 | 0.329 / 9.718 | 0.557 / 9.281 | 0.236 / 9.586 | 0.329 / 8.785 |
| **Taxonomy Parent** | **0.626** / **9.102** | **0.553** / **9.165** | **0.419** / **9.319** | **0.685** / **8.658** | **0.319** / **9.171** | **0.431** / **8.515** |

Key claim: only the prompt that states the *correct* hierarchical `is-a`
relations improves both Top-1 and cross-entropy on every dataset.

## Out of scope (appendix-only experiments)

* Table 7 (simulated Gaussian data, Appendix C)
* Table 8 (`D_ELCA`, Appendix D.3)
* Table 10 / Figure 8 (soft-label quality vs source-model LCA, Appendix E.4)

`D_ELCA` is nevertheless implemented in `lca_on_the_line.lca.dataset_elca`
(plus an un-normalised variant) for completeness.
