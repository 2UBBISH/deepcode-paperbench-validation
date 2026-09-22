# SMM — Sample-specific Multi-channel Masks for Visual Reprogramming (ICML 2024)

Reproduction of **"SMM: Sample-specific Multi-channel Masks for Visual Reprogramming"**.

Visual Reprogramming (VR) reuses a **frozen, pre-trained** ImageNet classifier for a new
target task by learning an input-space transformation `f_in`; the label space is bridged by a
non-parametric output mapping `f_out`. Classic VR learns a *single shared* perturbation pattern
for all inputs. **SMM** replaces the all-one / fixed mask with a *sample-specific, multi-channel*
mask produced by a tiny CNN, giving

```
f_in(x_i; δ, φ) = r(x_i) + δ ⊙ f_mask(r(x_i))            (paper Eq. 4)
```

where

* `r(·)`   — the paper's image transform (resize / crop / flip / RGB / ToTensor / ImageNet-normalize),
* `δ`      — the **shared** learnable pattern, initialized to **all zeros** before training (Algorithm 1),
* `f_mask` — a lightweight CNN (`φ`) whose output is a low-resolution 3-channel mask,
* `⊙`      — element-wise (pixel-wise) multiplication.

The low-resolution mask is expanded back to the input resolution by **patch-wise interpolation**
(pure block replication, `2^l × 2^l` per pixel, default `l = 3` ⇒ patch size 8), which the paper
shows is cheaper than floating-point interpolation while keeping gradients flowing to `φ`.

Only `δ` and `φ` are trained (≈ 0.54 M extra parameters for ViT-L; the mask generator alone is
26,499 parameters for ResNets and 102,339 for ViT-B/32). The pre-trained backbone is frozen.

---

## 1. Repository layout

```
smm_vr/
  configs/            YAML configuration layers
    base.yaml             global defaults (data, training, evaluation, output)
    datasets.yaml         Table 6 dataset registry + ordering manifests
    model_resnet18.yaml   224×224, 5-layer f_mask (26,499 params), α=0.01/γ=0.1
    model_resnet50.yaml   same mask generator as ResNet-18
    model_vit_b32.yaml    384×384, 6-layer f_mask (102,339 params), α=0.001/γ=1.0
    train_smm.yaml        main runs (Tables 1 & 2)
    ablations.yaml        Table 3 masking study + Figure 4 patch-size sweep
    label_mappings.yaml   Table 10 label-mapping study
  data/               transforms, datasets, splits, dataset_stats (Table 6 reference)
  models/             pretrained.py (frozen f_P), mask_generator.py (f_mask)
  modules/            patch_interp.py, reprogram.py (δ, f_in)
  label_mapping/      frequency.py (Alg. 2), flm.py (Alg. 3), ilm.py (Alg. 4), rlm.py
  engine/             train_smm.py (Alg. 1), evaluate.py, metrics.py, seeds.py
  methods/            baselines.py (Pad/Narrow/Medium/Full), finetuning.py (LoRA / FC)
  experiments/        run_main.py, run_ablations.py, run_label_mappings.py,
                      run_scaling.py, run_finetuning.py, run_stanfordcars.py
  analysis/           tsne_features.py, plot_patch_size.py
  scripts/            prepare_data.sh, run_all_main.sh, run_all_ablations.sh
  main.py             CLI dispatcher
  requirements.txt
```

---

## 2. Environment

Python 3.10+ on a single NVIDIA A100 is the reference setting; smaller GPUs work if you reduce
`--train-fraction` / batch sizes or run one dataset at a time. Install:

```bash
python -m pip install -r smm_vr/requirements.txt
```

Core pins: `torch>=2.0`, `torchvision>=0.15`, `numpy`, `scipy`, `scikit-learn` (t-SNE),
`pandas`, `pyyaml`, `matplotlib`, `tqdm`. Pre-trained weights (ImageNet-1K) are fetched by
`torchvision` on first use: ResNet-18, ResNet-50, ViT-B/32 (and optional ViT-L/16 for Appendix E).

---

## 3. Data preparation

All 11 main target tasks plus StanfordCars are torchvision datasets.

```bash
# downloads the 11 main datasets (+ StanfordCars) and prints Table 6 metadata
bash smm_vr/scripts/prepare_data.sh --data-root ./data

# or a subset
bash smm_vr/scripts/prepare_data.sh --data-root ./data --datasets "cifar10 svhn eurosat"
```

`prepare_data.sh` wraps `smm_vr.data.datasets.build_datasets` and falls back to plain
`torchvision.datasets` if the package is not importable yet. UCF101 may need the official
`UCF101.rar` placed under `<data-root>/UCF101` manually.

The data root can also be supplied by environment variable: `SMM_DATA_ROOT=./data`.

### Dataset registry (Table 6)

| key          | #classes | original | train | test  | split policy |
|--------------|---------:|---------:|------:|------:|--------------|
| cifar10      | 10       | 32       | 50000 | 10000 | native |
| cifar100     | 100      | 32       | 50000 | 10000 | native |
| svhn         | 10       | 32       | 73257 | 26032 | native |
| gtsrb        | 43       | 32       | 39209 | 12630 | native |
| flowers102   | 102      | 128      | 1020  | 6149  | ratio (class-balanced) |
| dtd          | 47       | 128      | 1880  | 1880  | ratio (class-balanced) |
| ucf101       | 101      | 128      | 9537  | 3783  | native (fold 1) |
| food101      | 101      | 128      | 75750 | 25250 | native |
| sun397       | 397      | 128      | 19850 | 19850 | ratio / torchvision split list |
| eurosat      | 10       | 128      | 13500 | 8100  | ratio (class-balanced) |
| oxfordpets   | 37       | 128      | 3680  | 3669  | ratio (class-balanced) |
| stanfordcars | 196      | 128      | 8144  | 8041  | native (failure case, Table 12) |

`data/splits.py` implements the deterministic class-balanced splitter for the `ratio` datasets,
so the split is identical across the three seeds **{0, 1, 2}** — only model/mask initialization
varies between seeds.

### Transforms (verbatim from the paper addendum)

```
imgsize = 384 if backbone == ViT_B32 else 224

train_preprocess = Compose([
    Resize((imgsize + 32, imgsize + 32)),
    RandomCrop(imgsize),
    RandomHorizontalFlip(),
    Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_preprocess = Compose([
    Resize((imgsize, imgsize)),
    Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

Note that `Resize` receives a 2-tuple, so aspect ratio is **not** preserved, and ImageNet
statistics are applied to every dataset (including the 32×32 ones) — both match the addendum.

---

## 4. Quick start

```bash
# list datasets / experiments
python -m smm_vr.main --list

# verify the environment and the Table 4 parameter budgets (26,499 / 102,339)
python -m smm_vr.main --verify

# show the merged config for a selection without running anything
python -m smm_vr.main --experiment main --backbone resnet18 --check-config --describe
```

### Main results — Tables 1 & 2

```bash
# Table 1: ResNet-18 and ResNet-50, SMM vs Pad/Narrow/Medium/Full
python -m smm_vr.main --experiment main --backbone resnet18 --datasets cifar10 --seeds 0

# Table 2: ViT-B/32
python -m smm_vr.main --experiment main --backbone vit_b32

# everything (all backbones, all 11 datasets, 3 seeds) — long
bash smm_vr/scripts/run_all_main.sh --mode all
```

### Ablations — Table 3 & Figure 4

```bash
python -m smm_vr.main --experiment ablations --mode masking      # Table 3
python -m smm_vr.main --experiment ablations --mode patch_size   # Figure 4 curves
python -m smm_vr.analysis.plot_patch_size --curves outputs/patch_size_curves.json \
        --output outputs/figures/figure4_patch_size.pdf

bash smm_vr/scripts/run_all_ablations.sh --mode all
```

### Additional studies

```bash
python -m smm_vr.experiments.run_label_mappings   # Table 10 (Rlm / Flm / Ilm)
python -m smm_vr.experiments.run_scaling          # Table 11 (f_mask capacity on EuroSAT)
python -m smm_vr.experiments.run_finetuning       # Tables 13 & 14 (LoRA / Finetuning-FC)
python -m smm_vr.experiments.run_stanfordcars     # Table 12 (failure case)
python -m smm_vr.analysis.tsne_features           # feature-space t-SNE (Figure 6 basis)
```

All runners accept `--seeds`, `--device`, `--data-root`, `--output-dir`, `--num-workers`,
`--train-fraction` and `--max-*-batches` for debugging.

---

## 5. Algorithms

| Paper | Module |
|-------|--------|
| Algorithm 1 — training loop | `engine/train_smm.py` (`train_smm`, `train_one_seed`, `train_with_seeds`) |
| Algorithm 2 — frequency matrix `d ∈ Z^{|Y^P| × |Y^T|}` | `label_mapping/frequency.py` |
| Algorithm 3 — Flm (frequent label mapping) | `label_mapping/flm.py` |
| Algorithm 4 — Ilm (iterated label mapping) | `label_mapping/ilm.py` |
| Rlm (random injective mapping) | `label_mapping/rlm.py` |

* **Rlm** samples `|Y^T|` ImageNet labels **without replacement** once, before training, and fixes them.
* **Flm** counts, per target class, the ImageNet argmax prediction of `f_P(f_in(x_i))` (identity
  `f_in`, i.e. `θ ← 0`), then greedily takes the largest entry of the frequency matrix, assigns it,
  and zeroes the selected row **and** column to preserve injectivity.
* **Ilm** repeats the Flm construction, recomputing `d` with the **current** `f_in` at the start of
  **every epoch**. It is the default mapping for the main tables.

All three mappings are non-parametric (integer index buffers, no learnable weights) and injective:
`f_out(y1) ≠ f_out(y2)` whenever `y1 ≠ y2`.

---

## 6. Reproduction targets

Top-1 test accuracy (%), mean over three seeds. Reference values from the paper.

### Table 1 — ResNet-18 / ResNet-50 (backbone frozen)

| method | ResNet-18 | ResNet-50 |
|--------|----------:|----------:|
| Pad    | 43.91 | 49.15 |
| Narrow | 43.48 | 46.76 |
| Medium | 45.04 | 49.39 |
| Full   | 46.85 | 52.10 |
| **SMM (ours)** | **52.53** | **56.35** |

### Table 2 — ViT-B/32

| method | avg |
|--------|----:|
| Pad    | 53.1 |
| Narrow | 63.7 |
| Medium | 65.2 |
| Full   | 64.7 |
| **SMM (ours)** | **72.4** |

Notable per-dataset SMM values: CIFAR-10 97.4, CIFAR-100 82.6, Flowers102 79.1, Food101 64.8.

### Table 3 — Impact of masking (ResNet-18, all 11 datasets)

| variant | avg |
|---------|----:|
| only δ (all-one mask) | 46.85 |
| only `f_mask` (no δ) | 42.59 |
| single-channel `f_mask^s` | 49.70 |
| **full SMM** | **52.53** |

### Figure 4 — Impact of patch size

Sweep `l ∈ {0,1,2,3,4}` (patch sizes 1, 2, 4, 8, 16) with ResNet-18. Accuracy improves from
constant masks, peaks around **patch size 8**, then plateaus/declines. 8 is also the default used
for every dataset.

### Table 10 — Label mapping study

| mapping | SMM improvement (avg points) |
|---------|------------------------------|
| Rlm | +8.94 |
| Flm | +4.69 |
| Ilm | +5.68 |

SMM improves all three mappings; Ilm is the best absolute performer.

### Table 11 — Capacity of `f_mask` (EuroSAT + ResNet-18)

Progressively double the intermediate channels of `f_mask`; the parameter ladder starts at the
26,499-parameter 5-layer generator. Test accuracy **peaks at a medium size** and stops improving
for very large generators (over-fitting), while train accuracy keeps rising.

### Tables 13 / 14 — Finetuning comparisons

* Table 13: LoRA for ViT-L (rank 6, LR 0.01, 10 epochs, ≈ 0.60 M extra params) vs SMM (≈ 0.54 M)
  on the four 32×32 tasks; SMM is favoured for these low-resolution tasks.
* Table 14: Finetuning-FC (ResNet-50) average accuracy improves **75.3 → 79.2** when the SMM input
  module is added, showing SMM is orthogonal to finetuning.

### Table 12 — StanfordCars failure case

Fine-grained 196-class StanfordCars: **all** methods (Pad/Narrow/Medium/Full/SMM) stay **below
10%** accuracy. SMM does not rescue a task on which VR itself fails.

### Feature-space t-SNE

Using **5000 randomly selected training samples per dataset** (`analysis/tsne_features.py`),
project the frozen classifier's output-layer features (taken *before* label mapping) to 2-D with
t-SNE. Class separation should look better for SMM than for the shared-mask baselines.

---

## 7. Configuration

YAML layers are merged left-to-right, later files win:

```
base.yaml → datasets.yaml → model_<backbone>.yaml → <experiment>.yaml
```

`main.py` performs this merge automatically (`resolve_config`) and passes the result to the
experiment runner. Key blocks:

* `model.*` — backbone, weights tag (`IMAGENET1K_V1`), `freeze: true`, `input_size` (224 ResNets /
  384 ViT-B/32), `num_pretrained_classes: 1000`, `feature_layer` (`avgpool` / `heads.head`).
* `mask_generator.*` — `num_layers` (5 ResNets / 6 ViT-B/32), `base_channels: 8`, `out_channels: 3`,
  `num_pooling_layers: 3`, `use_bn: true`, `use_relu: true`, `width_scale` (drives Table 11),
  `expected_parameters` (26,499 / 102,339 assertion targets).
* `reprogram.*` — `patch_size: 8`, `delta_init: zero`, `normalize_mask: false`.
* `training.*` — epochs, milestones, two learning rates (δ and φ), optimizer, batch sizes.
* `evaluation.*` — `topk`, `tsne_samples: 5000`.
* `output.*` — results / checkpoints / logs / figures directories.

---

## 8. Training protocol (Table 9 + paper defaults)

* **Epochs**: 200; **LR milestones**: 100 and 145 (the leading `0` printed in Table 9 is treated as
  a typo/artifact).
* **δ (shared pattern)**: `alpha_1 = 0.01`, decay `gamma = 0.1`, initialized to all zeros.
* **φ (mask generator)**:
  * 5-layer (ResNet-18 / ResNet-50): `alpha_2 = 0.01`, `gamma = 0.1`;
  * 6-layer (ViT-B/32): `alpha_2 = 0.001`, `gamma = 1.0` (no decay).
* **Batch size**: 256 for all datasets except **DTD** and **OxfordPets**, which use 64.
* Two independent optimizers + `MultiStepLR` schedulers (one for δ, one for φ).
* **Loss**: cross-entropy over the mapped target labels (the paper only says "classification loss").
* **Seeds**: {0, 1, 2}; results are reported as mean ± sample std (`ddof=1`).
* Checkpoint selection is by best test top-1 during training; the reported number is the best/final
  test accuracy per run, aggregated over seeds.

Frozen backbones keep `requires_grad=False` **and** stay in `eval()` mode, but the forward graph is
still differentiable so gradients reach `δ` and `φ` through the reprogrammed input.

---

## 9. Defaults chosen for paper ambiguities

Where the paper is silent, the following sensible defaults were adopted (also recorded in
`configs/*.yaml` comments and in `main.py --describe`):

1. **Optimizer** — SGD, momentum 0.9, `weight_decay = 0`, `nesterov = false`, no gradient clipping.
2. **Mask-generator channels** — geometric widths from base width 8 (`[8,16,32,64]` for 5 layers,
   `[8,16,32,64,128]` for 6), chosen so the parameter counts match Table 4 **exactly**
   (26,499 / 102,339). BatchNorm is enabled precisely because its affine parameters are required to
   hit those budgets.
3. **Activations** — ReLU after each hidden convolution; the final 3-channel convolution is linear.
4. **Mask output** — raw (not squashed) mask values; no sigmoid/softmax normalization.
5. **Loss** — cross-entropy over the mapped target-label logits.
6. **Milestones** — `[100, 145]`.
7. **Batch sizes** — 256 (64 for DTD / OxfordPets).
8. **Seeds** — `{0, 1, 2}`.
9. **Determinism** — deterministic cuDNN, `warn_only=True` (cheap: only the tiny mask generator
   trains over a frozen backbone).
10. **Split policy** — native torchvision splits where they exist; otherwise a deterministic
    class-balanced ratio split approximating Table 6 sizes (shared across seeds).

---

## 10. Known exceptions / caveats

* **ResNet-18 on DTD** — the shared-mask **Pad** baseline can slightly beat SMM; this is a known
  abnormal case, not a bug.
* **ViT-B/32 on EuroSAT** — **Pad** may be marginally better because SMM tends to over-fit this
  small 10-class, 13,500-image dataset.
* **UCF101** — sensitive to the learning-rate schedule; keep the paper's 200-epoch / milestones
  100,145 schedule, and treat any large deviation as a schedule issue rather than a code error.
* **StanfordCars** — all methods stay below 10% (VR simply fails on this fine-grained task);
  SMM is not expected to help.
* **Small debug runs** (`--train-fraction`, `--max-train-batches`) will not reach the numbers in
  the tables; they are for smoke-testing the pipeline only.

---

## 11. Validation checklist

Run these to confirm the implementation is faithful:

```bash
python -m smm_vr.main --verify
```

* [ ] Mask generator parameter counts: **26,499** (5-layer, ResNet-18/50) and **102,339** (6-layer, ViT-B/32).
* [ ] `δ` initializes to an all-zero tensor before training.
* [ ] `f_in(x) = r(x) + δ ⊙ mask` has the same shape as the pre-trained model input.
* [ ] Patch-wise interpolation at `l = 3` maps `28×28 → 224×224` and `48×48 → 384×384`.
* [ ] Rlm / Flm / Ilm mappings are injective (`f_out(y1) ≠ f_out(y2)` for `y1 ≠ y2`).
* [ ] Ilm is recomputed every epoch; Rlm and Flm are fixed.
* [ ] Average accuracies and the ordering of the methods roughly match Tables 1, 2, 3, 10.

Success is defined as reproducing the *averaged* results and the *ordering* of the tables, with the
documented exceptions above.

---

## 12. Out of scope

Per the reproduction plan, the following are intentionally **not** implemented: Figures 1, 2 and 6
as rendered in the paper, and the qualitative mask / shared-pattern visualization subsection.
(`analysis/tsne_features.py` still provides the numeric t-SNE basis for the feature-space section.)

---

## 13. Citation

```
@inproceedings{smm2024visual,
  title     = {SMM: Sample-specific Multi-channel Masks for Visual Reprogramming},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024}
}
```
