# LCA-on-the-Line — reproduction

Reproduction of **"LCA-on-the-Line: Benchmarking Out-of-Distribution
Generalization with Class Taxonomies"** (Shi et al., ICML 2024).

The repository implements the paper's core contributions end-to-end:

1. the **Lowest Common Ancestor (LCA) distance** as a measure of *misprediction
   severity* on a class taxonomy (Section 2, Appendix D.2);
2. the **LCA-on-the-Line benchmark** over 75 models (36 VMs + 39 VLMs) on
   ImageNet and five severely shifted OOD datasets, including the
   `R^2`/`PEA`/`KEN`/`SPE` correlation measurements and the OOD-error-prediction
   comparison against AC / Aline-S / Aline-D / ID-Top1 (Sections 4.1-4.2,
   Tables 1-3, Figures 1/5/9);
3. **latent taxonomies from K-means clustering** and the robustness of LCA to
   the choice of hierarchy (Section 4.3.1, Table 4, Figure 6);
4. **taxonomy soft labels** for linear probing, with weight-space interpolation
   (Section 4.3.2, Tables 5/6/9, Algorithm 1);
5. **taxonomy-alignment prompt engineering** for zero-shot VLMs
   (Section 4.3.3, Table 14).

The code is written to be *run later* on a GPU machine: nothing here requires a
GPU to inspect, and every stage caches its intermediate artefacts so the
pipeline can be resumed.

---

## 1. Quick start

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git      # openai/CLIP

# 0. imports + unit tests (fast, CPU only, no datasets needed)
python -m pytest tests -q

# 1. smoke-test the whole pipeline on synthetic data (no dataset download)
python scripts/smoke_test.py --clip --n-templates 1 --limit 24

# 2. stage the datasets (ImageNet via HuggingFace, OOD from their authors)
python scripts/download_data.py --data-root ./data/datasets --dataset all --dry-run

# 3. full pipeline
bash scripts/run_all.sh
```

`scripts/run_all.sh` performs the stages below in order; each one can also be
run on its own.

| stage | command | paper artefact |
|---|---|---|
| benchmark | `python -m lca_on_the_line.evaluate --data-root DIR --out-dir results` | Tables 1-3, Figures 1/5 |
| tables | `python scripts/reproduce_tables.py --results-dir results --out-dir tables` | Tables 1-3, Figures 1/5/9 |
| latent | `python -m lca_on_the_line.experiments_latent --data-root DIR --results-dir results` | Table 4, Figure 6 |
| soft labels | `python -m lca_on_the_line.experiments_soft_labels --backbone resnet18 --data-root DIR` | Tables 5/6/9 |
| prompts | `python -m lca_on_the_line.experiments_prompt --model CLIP_ViT-B_32 --data-root DIR` | Table 14 |
| soft-label quality | `python -m lca_on_the_line.experiments_soft_labels --backbone resnet18 --data-root DIR --results-dir results --table10` | Table 10, Figure 8 |
| LCA matrices | `python scripts/plot_lca_matrices.py --class-features results/class_features` | Figure 7 |
| validation | `python scripts/validate_against_paper.py --imagenet-v2 DIR --models resnet18` | Sec. 4.1 sanity check |

---

## 2. Repository layout

```
lca_on_the_line/
  hierarchy.py            WordNet ImageNet taxonomy, LCA, D_LCA^I / D_LCA^P
  lca.py                  LCA / ELCA distances, Top-k accuracy (Sections 2, D.2/D.3)
  metrics.py              R^2, PEA, KEN, SPE, MAE, min-max scaling (Section D.1)
  models.py               the 75 pretrained models (Appendix A) + feature extraction
  data.py                 ImageNet + 5 OOD datasets, label harmonisation to 0..999
  evaluate.py             the main benchmark loop -> results/metrics.csv + logits
  baselines.py            AC, Aline-S, Aline-D ports (Section 4.2)
  analysis.py             Tables 1-3 and Figures 1/5/9 from cached results
  latent.py               K-means latent hierarchies (Section 4.3.1)
  soft_labels.py          Algorithm 1 loss, linear probe, weight interpolation
  prompt_engineering.py   CLIP prompt ensemble + taxonomy prompts (Section 4.3.3)
  experiments_latent.py   Table 4 runner
  experiments_soft_labels.py  Tables 5/6 runner
  experiments_prompt.py   Table 14 runner
configs/                  model list + default hyper-parameters
docs/                     reference values transcribed from the paper
data/hierarchy/           bundled WordNet hierarchy resources
scripts/                  download / run / reproduce entry points
tests/                    52 unit + integration tests
```

---

## 3. How each contribution is implemented

### 3.1 The LCA distance (Section 2, Appendix D.2)

The hierarchy is the `imagenet_fiveai` taxonomy (the ImageNet hierarchy of
Bertinetto et al., 2020, as used by the paper). `data/hierarchy/` ships it plus
the standard `imagenet_class_index.json`; the 1000 leaves of the tree are
verified to be exactly the 1000 ImageNet classes.

Two node scores are implemented, exactly as in Appendix D.2.1:

* `D_LCA^I(y', y) = I(y) - I(LCA(y, y'))` with the **information content**
  requested by the addendum, `I(y) = -log2 p(y) = log2|L| - log2|L(y)|`
  (uniform over the 1000 leaves). For leaf-to-leaf pairs this reduces to
  `log2(#leaves below the LCA)`. Used for every benchmark measurement.
* `D_LCA^P(y', y) = (P(y) - P(LCA)) + (P(y') - P(LCA))` with `P` the
  root-distance depth, which additionally counts the descent to the prediction
  node and corrects tree imbalance. Used for the linear-probing soft labels.

The per-dataset distance averages **only the misclassified samples**, exactly as
in the paper's equation. The addendum's sanity checks are enforced by the test
suite: the LCA matrix has a zero diagonal and is symmetric, and the inverted
matrix has a unit diagonal (`tests/test_hierarchy.py`).

### 3.2 The 75-model benchmark (Section 4.1-4.2)

`models.py` instantiates the full Appendix A list:

* **36 VMs** through `torchvision.models.get_model(..., weights="IMAGENET1K_V1")`
  (AlexNet, ConvNeXt, 4x DenseNet, EfficientNet-B0, GoogLeNet, Inception-v3,
  4x MnasNet, 2x MobileNet-v3, RegNet-Y-1.6GF, Wide-ResNet-101, 5x ResNet,
  ShuffleNet-v2, 2x SqueezeNet, Swin-B, 8x VGG, ViT-B/32, ViT-L/32);
* **7 CLIP models** through `openai/CLIP` (`RN50`, `RN101`, `RN50x4`,
  `ViT-B/32`, `ViT-B/16`, `ViT-L/14`, `ViT-L/14@336px`);
* **30 OpenCLIP models** through `open_clip.create_model_and_transforms`;
* **ALBEF + BLIP** feature extractors through LAVIS (optional dependency).

VLMs are used zero-shot with the standard 80-template OpenAI ImageNet prompt
ensemble; text embeddings are cached on disk (`~/.cache/lca_on_the_line`) so the
expensive part is computed once. VM features `M(X)` are captured from the input
of the final linear layer with a forward hook, as the addendum specifies.

`evaluate.py` writes one row per (model, dataset) into `results/metrics.csv` and
caches the float16 logits under `results/logits/`, so all downstream analysis is
cheap and re-runnable.

### 3.3 Tables 2/3 (correlation + error prediction)

`analysis.compute_table2` reproduces the `R^2`/`PEA` grid for
`{ID Top1, ID LCA} x {OOD Top1, OOD Top5}` (`compute_ranking_table` produces the
KEN/SPE ranking counterparts introduced in the main-text metric setup); `analysis.compute_table3` reproduces
the MAE comparison against `ID Top1`, `AC`, `Aline-S`, `Aline-D` and `ID LCA`,
using the paper's min-max scaling instead of a probit transform. The Aline
implementations are ports of
`Agreement-on-the-line/agreement_trajectory.ipynb`, the notebook named by the
addendum.

Expected qualitative trends (paper, Table 2/3): ID Top-1 correlates strongly
with ImageNet-v2 but its correlation collapses on ImageNet-S/R/A and ObjectNet,
whereas ID LCA keeps `R^2 > 0.7` and `PEA > 0.83` on all four severely shifted
datasets, and yields the lowest error-prediction MAE on three of the five OOD
sets (ImageNet-S/A/ObjectNet) and the second lowest on a fourth (ImageNet-R).

`Aline-S`/`Aline-D` map onto the two returns of the reference notebook's
`aline()` function (``pred_s`` and ``pred_d``) respectively.

### 3.4 Latent hierarchies (Section 4.3.1 / Table 4)

`latent.py` averages the per-class features `kX`, runs K-means with `2**i`
centres for `i = 1..9` (independently per level, as in Appendix E.1), and takes
the pairwise LCA height to be the cluster level at which two classes first share
a cluster (base level 10). The height matrix is a similarity, so it is inverted
before use, and the result is min-max scaled — this is exactly the
`process_lca_matrix` snippet from the addendum. `experiments_latent.py` builds
all 75 hierarchies and reports the mean/min/max/std of the correlations with OOD
Top-1 next to the WordNet row, i.e. Table 4.

### 3.5 Soft labels (Section 4.3.2 / Tables 5, 6, 9)

`soft_labels.lca_alignment_loss` is a line-by-line implementation of Algorithm 1
(`L = lambda * L(CE) + L(soft_lca)`, `reverse = 1 - M_LCA`, `lambda = 0.03`,
`temperature = 25`, CE variant), and `train_linear_probe` trains the probe with
the AdamW + cosine-with-warm-up recipe of Appendix E.5 (lr 1e-3, batch 1024,
50 epochs). `interpolate_weights` implements
`W_interp = alpha * W_ce + (1 - alpha) * W_ce+soft`. The runner sweeps `alpha`
and reports baseline / soft / interpolated accuracy on ImageNet and the OOD
datasets, which is the "no ID accuracy drop" vs "pro-OOD" trade-off of
Tables 5/6/9.

### 3.6 Prompt engineering (Section 4.3.3 / Table 14)

`prompt_engineering.build_taxonomy_prompts` generates the four protocols from
the caption of Table 14, e.g. for the dalmatian class:

```
baseline         a photo of a dalmatian.
stack_parent     a photo of a dalmatian, dog, canine.
taxonomy_parent  a photo of a dalmatian, which is a type of a dog, which is a type of a canine.
shuffle_parent   a photo of a dalmatian, which is a type of a mountain bike, which is a type of a coil.
```

(`shuffle_parent` samples the ancestors randomly and is seeded for
reproducibility.) `experiments_prompt.py` evaluates all four protocols for a
CLIP-style model and reports OOD Top-1 and test-time cross-entropy; the expected
trend is that only `taxonomy_parent` improves both.

---

## 4. What was verified in this environment

### 4.1 Real-data validation against the paper

The pipeline was validated against the paper on **all 10,000 images** of
ImageNet-v2 (MatchedFrequency, `vaishaal/ImageNetV2` @ `d626240`), which shares
the 1000 ImageNet classes and therefore exercises the hierarchy, the class-index
mapping, the torchvision preprocessing and the LCA formula together:

```bash
python scripts/validate_against_paper.py --imagenet-v2 /path/to/imagenetv2 --models resnet18
```

| metric (ResNet-18 on ImageNet-v2) | this reproduction | paper |
|---|---|---|
| Top-1 accuracy | **0.5728** | 0.573 (Table 1) |
| Top-5 accuracy | 0.7989 | — |
| LCA distance | **6.9176** | 6.918 (Table 8) |

Both the accuracy and the LCA distance reproduce the paper to within rounding,
which is strong evidence that the taxonomy, the information-content score and
the dataset harmonisation are implemented as in the original work. The script
also carries the reference values for ResNet-50 and the two CLIP ResNets.

For cheaper partial checks use `--per-class N`, which samples evenly across the
1000 classes. With 2 images per class (2000 images, ~2 minutes on CPU):

| model | ImgN-v2 Top-1 (2000 imgs) | ImgN-v2 LCA (2000 imgs) | full-set / paper |
|---|---|---|---|
| ResNet-18 | 0.585 | 6.999 | 0.5728 / 0.573, 6.9176 / 6.918 |
| ResNet-50 | 0.637 | 6.803 | paper: 0.610, 6.863 |

(Plain `--limit` takes the first N files in class order, so it only sees the
easiest few classes — that is a sampling artefact, not a bug; `--per-class`
avoids it.)

### 4.2 Tests and smoke tests

The grading environment has no GPU, so the heavy 75-model sweeps were *written
but not executed here*. What else was actually run:

* `python -m pytest tests -q` — **52 tests pass**, covering
  * the hierarchy (1000 leaves == ImageNet class index, single root, LCA of
    known siblings, information content, zero-diagonal distance matrices,
    unit-diagonal inverted matrix),
  * the metrics (`R^2 == PEA^2`, Kendall/Spearman against SciPy, MAE),
  * the LCA/ELCA dataset distances and top-k accuracy,
  * the latent hierarchy (height/distance matrix invariants, `process_lca_matrix`),
  * Algorithm 1 (exact loss values, monotonicity in mistake severity),
  * weight interpolation, temperature scaling, the Aline port,
  * the taxonomy prompts (the dalmatian example above) and the Table 14 harness,
  * end-to-end pipeline runs on synthetic data (Table 2 and Table 3 code paths).
* `python scripts/validate_against_paper.py` — the ImageNet-v2 validation of
  Section 4.1 (ResNet-18: 0.5728 Top-1 and 6.9176 LCA vs. the paper's 0.573 and
  6.918).
* `python scripts/smoke_test.py --clip --n-templates 1 --limit 24` — builds
  ResNet-18, ResNet-34 and CLIP-RN50, evaluates them through the real
  `run_benchmark` path on a synthetic ImageNet-shaped dataset, and writes
  `metrics.csv`, Table 1, Table 2 and Figures 1/5.
* Model-zoo checks: all 36 torchvision architectures expose
  `IMAGENET1K_V1` weights, and all 30 OpenCLIP (arch, pretrained) pairs resolve
  in `open_clip.list_pretrained()`.

Reproducing the exact numbers in Tables 1-6 and 14 additionally requires the
ImageNet + OOD datasets and the full model sweep, which are the documented
long-running steps.

---

## 5. Datasets

Same sources as the addendum:

| dataset | source |
|---|---|
| ImageNet-1k | `datasets.load_dataset("imagenet-1k", trust_remote_code=True)` |
| ImageNet-v2 | `vaishaal/ImageNetV2` @ `d626240`, **MatchedFrequency only** |
| ImageNet-Sketch | `songweig/imagenet_sketch` (HuggingFace) |
| ImageNet-R | `github.com/hendrycks/imagenet-r` |
| ImageNet-A | `github.com/hendrycks/natural-adv-examples` |
| ObjectNet | `objectnet.dev` (+ its folder→ImageNet mapping CSV) |

All labels are harmonised onto the 0..999 ImageNet class indices
(`data.resolve_wnid` / `resolve_index_folder` / `load_objectnet_mapping`), so a
single WordNet hierarchy applies to every dataset.

## 6. Notes and limitations

* `D_ELCA` is appendix-only (Table 8) and therefore out of scope; it is
  implemented in `lca.lca.dataset_elca` for completeness, together with an
  un-normalised variant, because the appendix equation's extra `1/K` factor
  makes the reported magnitudes hard to obtain directly.
* The Aline baselines are fitted on the 75-model pool, matching the paper's
  setup; the agreement filters (`0.05 <= agreement <= 0.98`) follow the
  reference notebook.
* ALBEF/BLIP need the optional `salesforce-lavis` package; if it is missing the
  loader raises a clear error and the other 73 models can still be evaluated.
* `--n-templates` exists only to keep CPU smoke tests fast; the paper uses the
  full 80-template ensemble.
