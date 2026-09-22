# FOA — Test-Time Model Adaptation with Only Forward Passes

Reproduction of **Forward-Optimization Adaptation (FOA)**: a backpropagation-free,
weight-frozen test-time adaptation method that optimizes a small *learnable input
prompt* online with CMA-ES using an unsupervised fitness (prediction entropy +
per-layer CLS activation-statistics discrepancy against source ID statistics),
together with a *back-to-source activation shifting* of the final-layer CLS feature.

The whole pipeline is **forward-only**: no gradients, no backward passes, no weight
updates. The only mutable objects at test time are

1. the CMA-ES search distribution over the flattened prompt vector `p ∈ R^{d·N_p}`,
2. the EMA state `μ_N(t)` used by the activation-shifting rule.

---

## 1. Repository layout

```
foa_repro/
  src/
    models/
      vit_loader.py          # frozen timm ViT-Base + all-layer [CLS] hooks
      prompt_injection.py    # learnable prompt, [CLS, prompts, patches] ordering
    method/
      foa.py                 # Algorithm 1 main loop (core)
      fitness.py             # Eqn. (5): entropy + activation discrepancy
      activation_shifting.py # Eqn. (7)-(9) + EMA state
      source_stats.py        # {mu_i^S, sigma_i^S} bank
      cma_wrapper.py         # CMA-ES ask/tell wrapper, Eqn. (6) init
      foa_interval.py        # FOA-I V1/V2 for BS=1 (Table 6)
    data/
      datasets.py            # ImageNet-1K / -C / -R / -V2(MF) / -Sketch streams
      corruption_stream.py   # non-i.i.d. streams (label-shift, mixed-shift)
    eval/
      metrics.py             # Accuracy, ECE (15 equal-width bins)
    baselines/
      lame.py t3a.py tent.py sar.py cotta.py memo.py
    quantization/
      ptq4vit_adapter.py     # 8/6-bit quantized ViT (PTQ4ViT)
    utils/
      config.py seeding.py logging_utils.py checkpoint_io.py
  scripts/
    compute_source_stats.py  # offline source statistics (Q=32)
    run_foa.py               # main ImageNet-C/R/V2/Sketch runner
    run_baselines.py         # LAME / T3A / TENT / SAR / CoTTA / MEMO / NoAdapt
    run_ablations.py         # Table 5 (components) + Table 9 (design choices)
    run_sensitivity.py       # Figure 2, Tables 13-15
    run_non_iid.py           # Table 11 (non-i.i.d. robustness)
  configs/
    foa_imagenetc.yaml foa_imagenetr.yaml foa_imagenetv2.yaml foa_sketch.yaml
    foa_quantized.yaml ablations.yaml sensitivity.yaml non_iid.yaml
  requirements.txt
```

---

## 2. Environment setup

Python **3.10+**, PyTorch ≥ 2.0 with CUDA (a single 16 GB GPU such as an RTX 3090 is
enough — FOA retains no activations for backward).

```bash
# (install the torch CUDA wheel for your platform first)
pip install -r requirements.txt
```

Key packages: `torch`, `torchvision`, `timm`, `datasets` + `huggingface_hub`, `cmaes`
(CyberAgentAILab; `pycma` works as a fallback and is auto-detected), `numpy`, `scipy`,
`pyyaml`, `tqdm`, `pandas`, `scikit-learn`, `Pillow`, optional `matplotlib`.

External code installed separately (per the addendum):

| Purpose | Repository |
| --- | --- |
| 8/6-bit quantization | https://github.com/hahnyuan/PTQ4ViT |
| LAME baseline | https://github.com/fiveai/LAME |
| T3A baseline | https://github.com/matsuolab/T3A |
| TENT baseline | https://github.com/DequanWang/tent |
| SAR baseline | https://github.com/mr-eggplant/SAR |
| CoTTA baseline | https://github.com/qinenergy/cotta |
| MEMO baseline (optional) | https://github.com/zhangmarvin/memo |

All baseline adapters in `src/baselines/` are self-contained reference
implementations with the paper's tuned hyper-parameters, so the comparison runs even
if the official repos are absent. `ptq4vit_adapter.py` uses the official PTQ4ViT when
importable and otherwise falls back to an equivalent twin-uniform min-MSE PTQ
emulation (32 calibration samples).

### Model weights

Download the timm ViT-Base augreg checkpoint once (URL is defined as
`FOA_CHECKPOINT_URL` in `src/models/vit_loader.py`):

```
B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz
```

It is loaded by `build_vit(...)`; set `model.checkpoint` in a config (or pass
`--checkpoint`) to use a local path.

### Data

* ImageNet-1K via HuggingFace: `load_dataset("imagenet-1k", trust_remote_code=True)`
  (used for the clean sanity check and for the `Q` source-statistics samples).
* ImageNet-C (severity 5, 15 corruptions), ImageNet-R, ImageNet-V2
  (matched-frequency subset), ImageNet-Sketch as local directories under `data/`
  (see `data.root` / `source_stats.id_root` in the configs).

---

## 3. Reproducing the paper

### Step 0 — source ID statistics (once)

Offline forward pass over `Q = 32` unlabeled ImageNet-1K validation images **without**
prompt injection; stores `{μ_i^S, σ_i^S}_{i=0..N}` and `μ_N^S`.

```bash
python scripts/compute_source_stats.py \
    --config configs/foa_imagenetc.yaml \
    --num-samples 32 \
    --output checkpoints/source_stats_vit_base.pt
```

Sanity check: feeding in-distribution images back should give an activation
discrepancy term near zero.

### Step 1 — backbone sanity (NoAdapt)

```bash
# clean ImageNet-1K accuracy should be ~85.17%
python scripts/run_foa.py --config configs/foa_imagenetc.yaml \
    --dataset imagenet-1k --limit-batches 20

# ImageNet-C Gaussian level 5: expect ~56.8% acc / ~7.5% ECE for NoAdapt
python scripts/run_baselines.py --config configs/foa_imagenetc.yaml \
    --methods noadapt --corruptions gaussian_noise
```

Averaged NoAdapt over the 15 corruptions should be ~55.5% acc / ~10.5% ECE.
Deviations mean the checkpoint, preprocessing or head mapping is wrong.

### Step 2 — main full-precision results

```bash
# ImageNet-C, all 15 corruptions (Table 2 / Table 16): FOA 66.3 acc / 3.2 ECE
python scripts/run_foa.py --config configs/foa_imagenetc.yaml --all-corruptions

# ImageNet-R (lambda_base = 0.2), ImageNet-V2 (MF), ImageNet-Sketch (Table 3)
python scripts/run_foa.py --config configs/foa_imagenetr.yaml
python scripts/run_foa.py --config configs/foa_imagenetv2.yaml
python scripts/run_foa.py --config configs/foa_sketch.yaml
```

Paper targets: **R 63.8 / V2 75.4 / Sketch 49.9 (avg 63.0 acc, 4.6 ECE)**.

### Step 3 — baselines (Table 2 comparison)

```bash
python scripts/run_baselines.py --config configs/foa_imagenetc.yaml \
    --methods noadapt lame t3a tent sar cotta --all-corruptions
```

Appendix B.2 hyper-parameters baked into the adapters:

| Method | Hyper-parameters |
| --- | --- |
| NoAdapt | frozen backbone, source softmax |
| LAME | kNN k=5, cosine affinity, BS=64 |
| T3A | M=20 supports/class, K=50, BS=64 |
| TENT | SGD, momentum 0.9, lr 1e-3, BS=64, LayerNorm affine params |
| SAR | SGD, momentum 0.9, lr 1e-3, entropy threshold `0.4·ln C`, blocks 1–8 norm affine |
| CoTTA | SGD, momentum 0.9, lr 0.05, threshold 0.1, 32 augmentations, restoration 0.01, teacher EMA 0.999 |
| MEMO | SGD over 32 augmented views, episodic per-sample adaptation |

Paper ImageNet-C: **FOA 66.3/3.2**, SAR 62.7/7.0, CoTTA 61.7/6.5, TENT 59.6/18.5,
T3A 56.9/26.8, LAME 54.1/11.0, NoAdapt 55.5/10.5.

### Step 4 — component ablations (Table 5)

```bash
python scripts/run_ablations.py --config configs/ablations.yaml --groups components
```

Expected ordering (15-corruption averages, ViT-Base, ImageNet-C level 5):

| Variant | Acc | ECE |
| --- | --- | --- |
| NoAdapt | 55.5 | 10.5 |
| CMA + entropy only | ≈44.9 | 36.8 (degrades below NoAdapt — the paper's motivation) |
| CMA + activation discrepancy | 63.4 | 9.4 |
| Activation shifting only | 59.1 | 12.7 |
| Entropy + discrepancy, no shifting | 65.4 | 3.3 |
| **Full FOA** | **66.3** | **3.2** |

### Step 5 — design-choice study (Table 9)

```bash
python scripts/run_ablations.py --config configs/ablations.yaml --groups design
```

Highlights: `norm_affine + CMA` collapses to ~0.1% accuracy (exp2/exp3), while
`prompts + CMA + Eqn.(5)` reaches 65.4/3.3; `norm_affine + SGD + Eqn.(5)` reaches
70.5/7.9 — showing Eqn. (5) also helps gradient training. The SGD + Eqn.(5) variant
divides the entropy term by BS=64 and uses `λ = 30`; the prompt + SGD + entropy
variant uses `lr = 0.01` with 3 prompts.

### Step 6 — quantized models (Table 4, Table 17)

```bash
python scripts/run_foa.py --config configs/foa_quantized.yaml --bits 8
python scripts/run_foa.py --config configs/foa_quantized.yaml --bits 6
```

Calibration uses 32 randomly selected ImageNet-1K training samples. Headline claim:
**8-bit FOA 63.5% / 3.8% ECE** and **6-bit FOA 55.8% / 5.5% ECE** on ImageNet-C level 5,
with 8-bit FOA (63.5%) beating 32-bit TENT (59.6%). The FOA loop itself is unchanged
because it never calls `backward()`.

### Step 7 — sensitivity (Figure 2, Tables 13–15)

```bash
python scripts/run_sensitivity.py --config configs/sensitivity.yaml \
    --groups population_size num_prompts source_samples lambda shifting_ema discrepancy_ema
```

Expected behaviour: accuracy converges for `K > 15` (K=2 already beats T3A, K=6 beats
TENT); stable for `Q ≥ 32` (Q ∈ {16, 32, 64, 100, 200, 400, 800, 1600}); small
variation for `N_p ∈ {1..10}`; `λ ∈ {0.3, 0.4, 0.5}` best and overall insensitive.

### Step 8 — FOA-I interval adaptation for BS = 1 (Table 6, Table 7)

```bash
python -m src.method.foa_interval --config configs/foa_imagenetc.yaml \
    --intervals 4 8 16 32 64 --variant v1   # and --variant v2
```

`V1` caches all-layer CLS features of buffered samples; `V2` caches raw images and
recomputes features at the update step. `I = 4` gives ≈62.1% on Gaussian level 5
versus 60.3% for TENT at BS=64.

### Step 9 — non-i.i.d. robustness (Table 11)

```bash
python scripts/run_non_iid.py --config configs/non_iid.yaml \
    --scenarios mild label_shift mixed_shift --methods foa
```

Targets: FOA 62.1/6.6 under label shift and 62.0/4.9 under mixed shift, compared with
TENT/SAR.

---

## 4. Algorithm summary (what the code implements)

Per test batch `X_t` (Algorithm 1):

1. **Sample** `K` candidate prompts from the CMA-ES distribution
   `p_k = m + τ · Σ^{1/2} · N(0, I)` (Eqn. 6), with `m⁰ = 0`, `τ⁰ = 1.0`, `Σ⁰ = I`
   and `K = 4 + 3·ln(d·N_p) = 28` for `d·N_p = 2304`.
2. **Forward** the frozen ViT for each candidate with the prompt injected as
   `[CLS, prompts, patches]`, collecting all-layer final CLS tokens `{e_n^0}`.
3. **Activation shifting** (Eqn. 7–9): `d_t = μ_N^S − μ_N(t−1)`,
   `ê_N^0 = e_N^0 + γ·d_t` with `γ = 1.0`, applied **before** the head; then
   `μ_N(t) = α·μ_N(X_t) + (1−α)·μ_N(t−1)` with `α = 0.1` and `μ_N(0)` initialized from
   the first batch. The shift uses only the un-shifted features for the EMA update.
4. **Classify** with the frozen head and score each candidate with Eqn. (5):

   ```
   F = Σ_b H(σ(f(x_b)))  +  λ · Σ_{i=1..N} ( ||μ_i(X_t) − μ_i^S||_2 + ||σ_i(X_t) − σ_i^S||_2 )
   ```

   with `λ = 0.4 · BS/64` on ImageNet-C/V2/Sketch and `0.2 · BS/64` on ImageNet-R.
   Test-batch statistics use the current batch directly (`β = 1.0`).
5. **Tell** the CMA-ES optimizer the `K` fitness values, and emit the prediction of the
   **best-fitness candidate** for the batch (no averaging over candidates).

---

## 5. Configs and key hyper-parameters

| Key | Value | Meaning |
| --- | --- | --- |
| `prompt.num_prompts` | 3 | `N_p` (prompts prepended after [CLS]) |
| `prompt.init` / `init_range` | `uniform` / `0.01` | uniform init (range is a documented default) |
| `cma.population_size` | 28 | `K = 4 + 3·ln(d·N_p)` |
| `cma.sigma0` / `mean0` / `cov0` | 1.0 / 0.0 / 1.0 | `τ⁰`, `m⁰`, `Σ⁰ = I` |
| `fitness.lambda_base` | 0.4 (0.2 on R) | `λ`, scaled by `BS/64` |
| `fitness.beta` | 1.0 | batch statistics used directly |
| `fitness.layer_start` | 1 | Eqn. (5) sums layers `i = 1..N` |
| `shifting.gamma` / `alpha` | 1.0 / 0.1 | Eqn. (7)/(9) |
| `data.batch_size` | 64 | test batch size |
| `source_stats.num_samples` | 32 | `Q` source images |
| `eval.ece_bins` | 15 | equal-width ECE bins |
| `seed` | 0 | global reproducibility seed |

### Documented defaults for paper ambiguities

* Eqn. (5) sums **both** mean and std discrepancies over `i = 1..N` (layer 0 is stored
  for `μ_N^S`/shifting use but excluded from the sum).
* Prompt dim `d = 768` for ViT-Base, so the CMA search space is `768·3 = 2304` by default.
* Prompt uniform-init range is unspecified in the paper → `U(-0.01, 0.01)`.
* ECE binning is unspecified (the paper cites Naeini et al. 2015) → 15 equal-width bins
  (last bin right-closed); accuracy/ECE are reported in percent.
* Activation-shifting update order: initialize `μ_N(0)` from the first batch → shift →
  EMA update on un-shifted features.
* The 32 source images are drawn with a fixed seed from the ImageNet-1K validation
  split, and a single global seed is used for CMA sampling.

### Not reproduced (excluded by the addendum)

Run-time memory usage (Table 7), computational complexity (Table 8), and
in-distribution performance (Table 12).

---

## 6. Quick smoke test

```bash
python scripts/compute_source_stats.py --config configs/foa_imagenetc.yaml --quiet
python scripts/run_foa.py --config configs/foa_imagenetc.yaml \
    --corruptions gaussian_noise --limit-batches 4 --device cuda
python scripts/run_ablations.py --config configs/ablations.yaml \
    --groups components --limit-batches 4
```

All runners write JSON payloads with the resolved config, measured metrics and the
paper reference numbers to the configured `output_dir`, so measured-vs-paper deltas
are visible directly in the artifacts.

---

## 7. Success criteria

* Full-precision ImageNet-C within roughly ±1% accuracy and ±1% ECE of the reported
  numbers (FOA 66.3 / 3.2).
* Component ordering of Table 5 preserved: activation discrepancy ≫ entropy-only
  (which must underperform NoAdapt), and activation shifting adds on top.
* 8-bit FOA (63.5%) > 32-bit TENT (59.6%) on ImageNet-C level 5.
