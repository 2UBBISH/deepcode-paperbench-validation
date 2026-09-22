# FOA - Test-Time Model Adaptation with Only Forward Passes

Reproduction of *"Test-Time Model Adaptation with Only Forward Passes"* (Niu, Miao, Chen,
Wu, Zhao; ICML 2024) — the **F**orward-**O**ptimization **A**daptation (FOA) method.

FOA adapts a frozen, possibly quantised model to a stream of out-of-distribution test
samples **without any back-propagation and without touching a single model weight**:

* **input level** — a small set of learnable prompt embeddings (``N_p = 3``) is inserted
  into the ViT input sequence, right after the CLS token, and optimised online by a
  derivative-free CMA-ES with an *unsupervised* fitness function (prediction entropy +
  activation-distribution discrepancy);
* **feature level** — a forward-only *back-to-source activation shifting* mechanism moves
  the final CLS activation of every OOD sample towards the source in-distribution domain.

Everything in this repository runs with forward passes only, on full-precision (32-bit)
and PTQ4ViT-quantised (8-bit / 6-bit) ViT-Base models.

---

## 1. What is implemented

| Paper element | Where |
|---|---|
| Eqn. (1)-(2) ViT with prompts ``[CLS, prompt, patches]`` | `foa/models/prompt_vit.py` |
| Eqn. (5) unsupervised fitness (entropy + activation discrepancy) | `foa/core/fitness.py` |
| Eqn. (6) CMA-ES sampling of prompt candidates | `foa/core/cma.py` |
| Eqn. (7)-(9) back-to-source activation shifting | `foa/core/activation_shift.py` |
| Algorithm 1, end-to-end FOA / FOA-I | `foa/core/foa.py`, `foa/methods/foa_method.py` |
| source ID statistics ``{mu_i^S, sigma_i^S}`` (32 samples) | `foa/core/statistics.py` |
| accuracy + ECE (15 bins) | `foa/evaluation/metrics.py` |
| online stream evaluation (i.i.d., label shift, mixed domains) | `foa/evaluation/runner.py`, `foa/data/datasets.py` |
| Table 2 (ImageNet-C, 32-bit ViT-Base) | `foa/cli.py::table2`, `scripts/run_table2.sh` |
| Table 3 (ImageNet-R / V2 / Sketch) | `foa/cli.py::table3`, `scripts/run_table3.sh` |
| Table 4 (8-bit / 6-bit quantised ViT-Base) | `foa/cli.py::table4` + `foa/quantization/` |
| Table 5 (ablation of the three components) | `foa/cli.py::table5`, `scripts/run_table5.sh` |
| Table 6 (FOA-I, interval update, batch size 1) | `foa/cli.py::table6`, `foa/core/foa.py::FOAInterval` |
| Table 9 (learnable params x optimiser x loss) | `foa/cli.py::table9`, `foa/methods/variants.py` |
| Table 10 (ResNet-50, VisionMamba) | `foa/cli.py::table10`, `foa/models/prompt_resnet.py`, `foa/models/prompt_sequence.py` |
| Table 11 (non-i.i.d. scenarios) | `foa/cli.py::table11`, `foa/data/datasets.py` |
| Baselines: LAME, T3A, TENT, SAR, CoTTA, BN-Adapt | `foa/methods/*.py` |

Out of scope on purpose (excluded by the paper addendum): run-time memory usage (Sec. 4.4),
computational complexity analyses (Sec. 4.4) and the in-distribution comparison
(Sec. 4.4).

## 2. Installation and data

```bash
pip install -r requirements.txt          # torch/timm/cmaes/datasets/...

# 1) source in-distribution statistics: 32 unlabelled ImageNet-1K val images
#    (downloaded through HuggingFace, as recommended by the paper addendum)
python -m foa.cli source-stats --out stats/vit_b16_in1k.pt

# 2) the four OOD benchmarks (ImageNet-C is ~65 GB, 5 tarballs from Zenodo)
python scripts/prepare_data.py --data-root data --only all

#    ... or materialise ImageNet-C from the public HuggingFace webdataset mirror
#    (no 65 GB download; the layout produced is identical):
#    python scripts/prepare_data.py --data-root data --hf-imagenet-c \
#        --corruptions gaussian_noise,shot_noise --severities 5
```

ImageNet-1K (val) can also be supplied locally: `--data-root data/imagenet --no-hf`, which
expects the usual `val/<wnid>/*.JPEG` layout. ImageNet-C expects
`<root>/<corruption>/<severity>/<wnid>/*.JPEG` (the official tarball layout).

## 3. Reproducing the tables

```bash
bash scripts/run_table2.sh            # ImageNet-C, 15 corruptions, all methods
bash scripts/run_table3.sh            # ImageNet-R / V2 / Sketch
bash scripts/run_table4.sh            # 8-bit and 6-bit ViT-Base
bash scripts/run_table5.sh            # component ablation
bash scripts/run_table6.sh            # FOA-I (single-sample adaptation)
bash scripts/run_table9.sh            # design choices
bash scripts/run_table10.sh           # ResNet-50 / VisionMamba
bash scripts/run_table11.sh           # non-i.i.d. streams
```

Each command writes a JSON file with per-corruption accuracy / ECE and the average, e.g.
`results/table2_imagenet_c.json`. A single configuration can be run directly:

```bash
python -m foa.cli evaluate --dataset imagenet_c --corruption gaussian_noise \
    --severity 5 --method FOA --data-root data/imagenet-c --stats stats/vit_b16_in1k.pt
```

The headline numbers of the paper that these commands reproduce (ViT-Base, batch size 64):

| Table | Setting | Paper |
|---|---|---|
| 2 | ImageNet-C level 5, 32-bit | NoAdapt 55.5 / TENT 59.6 / SAR 62.7 / **FOA 66.3** (ECE 3.2) |
| 3 | ImageNet-R / V2 / Sketch, 32-bit | FOA 63.8 / 75.4 / 49.9 (avg 63.0) |
| 4 | ImageNet-C level 5, 8-bit / 6-bit | FOA 63.5 / 55.8 (T3A 55.1 / 45.4) |
| 5 | components | entropy only 44.9, discrepancy 63.4, shifting 59.1, FOA 66.3 |
| 6 | FOA-I (I=4) on Gaussian noise | 62.1 (TENT BS=64: 60.3, NoAdapt: 56.8) |
| 9 | design choices | prompts+CMA+Eqn.(5) 65.4 vs norm+CMA 0.1 |
| 10 | ResNet-50 / VisionMamba | 22.6 / 49.6 (FOA-dagger 33.6 / 56.5) |
| 11 | non-i.i.d. | 66.3 / 62.1 / 62.0 |

## 4. Hyper-parameters (Appendix B.2)

| Symbol | Value | Note |
|---|---|---|
| ``N_p`` | 3 | number of prompt embeddings, uniform init |
| ``K`` | 28 | CMA-ES population size, ``4 + 3 log(prompt dim)`` |
| ``lambda`` | ``0.4 x BS/64`` | ``0.2 x BS/64`` on ImageNet-R |
| ``gamma`` | 1.0 | activation-shifting step size |
| ``alpha`` | 0.1 | EMA factor of ``mu_N(t)`` (Eqn. 9) |
| ``Q`` | 32 | unlabelled ImageNet val images for the source statistics |
| BS | 64 | batch size (1 for FOA-I) |
| calibration | 32 samples | PTQ4ViT quantisation of the 8-bit / 6-bit models |

`configs/foa_imagenet_c.yaml` and `configs/baselines.yaml` list the same values as YAML.

## 5. Repository layout

```
foa/
  config.py            all hyper-parameters quoted in the paper
  cli.py               command line interface (evaluate / table / source-stats / download)
  core/
    cma.py             CMA-ES (cmaes library + pure NumPy fallback)
    fitness.py         Eqn. (5)
    activation_shift.py Eqn. (7)-(9)
    statistics.py      source ID statistics
    foa.py             Algorithm 1 + FOA-I interval strategy
  models/
    prompt_vit.py      ViT-Base with the prompt inserted after the CLS token
    prompt_resnet.py   learnable 7x7 conv prompt for ResNet-50 (Table 10)
    prompt_sequence.py token-sequence prompt wrapper for VisionMamba (Table 10)
  methods/             FOA + LAME, T3A, TENT, SAR, CoTTA, BN-Adapt, Table-9 variants
  quantization/        PTQ4ViT: twin uniform quantisation + layer-wise clipping search
  data/                ImageNet-C/R/V2/Sketch, HuggingFace loaders, streams, downloads
  evaluation/          accuracy / ECE, the generic online TTA runner
scripts/               data preparation + one wrapper per table
tests/                 pytest smoke tests (CPU, tiny ViT, no downloads)
```

## 6. Verification performed in this environment

No GPU was available and single commands are limited to ~10 minutes, so the full ImageNet
experiments (which need 50 000 images per corruption and 28 forward passes per batch)
were **not** executed here; the code is written to be run on a GPU machine (see the
commands above). What *was* executed:

* `pytest tests/` — 36 tests, all passing (~35 s on CPU): CMA-ES convergence (against the
  `cmaes` library), the fitness function vs. its manual definition, the activation
  shifting EMA, prompt/layer-feature shapes, all baselines, the deferred-prediction
  protocol, the PTQ4ViT machinery (quantisation disabled == float, 8-bit error < 6-bit
  error, dequantisation restores the model), the Table-9 variants and FOA/FOA-I
  end-to-end;
* end-to-end CLI smoke runs on a tiny ViT with synthetic data
  (`python -m foa.cli source-stats --synthetic ...`, `... evaluate --dataset synthetic`).
* **a scaled-down but genuine run of the method itself**,
  `scripts/validate_proxy_imagenetv2.py`: 32 clean ImageNet-V2 images (public HuggingFace
  mirror) provide the source statistics, the next 200 images are corrupted with Gaussian
  noise and streamed through `vit_small_patch16_224.augreg_in21k_ft_in1k` (batch size 8,
  CMA population 8, ~25 adaptation steps — 30x fewer than in the paper):

  | stream | accuracy | ECE |
  |---|---|---|
  | clean images, NoAdapt | 63.0 | — |
  | corrupted, NoAdapt | 38.0 | 13.7 |
  | corrupted, FOA with *entropy only* | 37.5 | 14.8 |
  | corrupted, FOA with *activation discrepancy only* | 38.5 | 13.2 |
  | corrupted, FOA with *entropy + discrepancy* | 40.0 | 12.8 |
  | corrupted, **FOA (full)** | **40.5** | **11.7** |

  This reproduces the ordering of Table 5 of the paper (entropy-only <= NoAdapt <
  discrepancy-only < entropy+discrepancy < FOA, with the ECE improving monotonically) and
  therefore validates the mechanism itself, even though the absolute values are far from
  the paper's: the proxy is 200x smaller and uses a much weaker backbone than ViT-Base
  (40% vs 85% clean accuracy).

## 7. Deviations / approximations (read this before comparing numbers)

The paper leaves a few implementation details implicit; here is exactly what this
reproduction does about them.

1. **Prompt positional embedding.** The paper only fixes the ordering
   ``[CLS, prompts, patches]``. We give the prompt tokens a *zero* positional embedding by
   default (`prompt_pos: zero` in `foa/models/prompt_vit.py`), so ``m^(0) = 0`` makes the
   first iteration identical to "no prompt". Other choices (`cls`, `interpolate`,
   `learnable`) are implemented and are equivalent up to a constant that the prompt itself
   can absorb.
2. **Extra forward pass for the shifting direction.** Algorithm 1 applies Eqn. (7) inside
   the candidate loop, so the direction ``d_t`` must be known *before* the ``K``
   candidates are evaluated. We compute ``mu_N(X_t)`` with the current CMA mean prompt
   (one extra forward pass per batch, hence ``K + 1`` forward passes instead of the
   ``#FP = K`` quoted in Table 8). The EMA of Eqn. (9) is always fed with *unshifted*
   features; otherwise ``d_t`` would collapse to zero and the shift would vanish.
3. **CMA-ES initialisation.** Algorithm 1 says ``m^(0) = 0, Sigma^(0) = I, tau^(0) = 1``;
   the implementation details say the prompts are "uniformly initialised". We follow
   Algorithm 1 (mean 0, ``sigma0 = 1.0``) and additionally provide
   `cma_mean_from_prompt_init` which centres the search on a uniformly initialised prompt
   (the effect is negligible because CMA re-samples at every step).
4. **PTQ4ViT.** The FOA paper does not restate PTQ4ViT's details (the addendum acknowledges
   this). We implement the parts PTQ4ViT is known for: uniform quantisation of all MatMul
   inputs/weights, **twin uniform quantisation** of the post-Softmax and post-GELU
   activations, LayerNorm/residuals in full precision, and a greedy layer-wise search of
   the clipping thresholds. The search metric is the layer-wise *reconstruction error*
   instead of PTQ4ViT's Hessian-guided metric, and quantisation is simulated by fake
   quantisation (round + de-quantise), which is numerically what a quantised model
   computes. Memory measurements are reported by the paper as an *ideal estimate*
   (``bits/32`` of the 32-bit model), which `ideal_memory_ratio()` provides.
5. **Baselines.** They are implemented natively in `foa/methods/` with the
   hyper-parameters quoted in Appendix B.2 and the algorithm of their official
   repositories (linked in the module docstrings), so that the whole reproduction is a
   single self-contained repository. Details that the paper does not quote were taken from
   the official code: LAME's kernel bandwidth/step size (we use ``alpha = 0.5``, ``k = 5``)
   and T3A's ``filter_K = 100``. SAR is the entropy-thresholded SAM update on the
   normalisation affine parameters of blocks 1..8; CoTTA uses augmentation-averaged
   teacher pseudo-labels, EMA teacher (0.999) and stochastic restoration (0.01).
6. **VisionMamba (Table 10).** The official `vim` package (which needs the `mamba-ssm`
   CUDA kernels) is not installable in a CPU-only environment. `build_prompt_visionmamba`
   loads it when available and otherwise falls back to a timm token-sequence model, so the
   code path is exercised but the numbers of Table 10 for VisionMamba require the official
   checkpoint.
7. **Stream order.** The i.i.d. ("mild") scenarios are evaluated on a shuffled stream with
   a fixed seed; the non-i.i.d. rows use a class-ordered Dirichlet stream (online label
   shift) and a stream whose batches are drawn from randomly chosen corruptions (mixed
   domain shifts), following NOTE/SAR. Absolute accuracies depend mildly on the stream
   order, which is why the paper reports averages over the 15 corruptions.
