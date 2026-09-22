# DPMs-ANT — reproduction

Reproduction of

> Xiyu Wang, Baijiong Lin, Daochang Liu, Ying-Cong Chen, Chang Xu.
> **Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer
> Learning.** ICML 2024.

The repository contains a complete implementation of the two contributions of
the paper — **similarity-guided training** (Section 4.1, Eq. 5) and
**adversarial noise selection** (Section 4.2, Eqs. 6–7) — the
parameter-efficient **adaptor module** (Section 4.3, Algorithm 1), the training
recipe of the paper's few-shot experiments (Section 5.2 + supplementary
material) and the evaluation metrics (Intra-LPIPS / FID, Section 5.2).

Nothing from the authors' own repository is used.

---

## 1. What is implemented

| Paper | Code |
|---|---|
| Section 3 — forward/reverse process, `sigma_hat` | [`dpms_ant/schedules.py`](dpms_ant/schedules.py), [`dpms_ant/sampling.py`](dpms_ant/sampling.py) |
| Eq. 5 — similarity-guided training loss | [`dpms_ant/guidance.py`](dpms_ant/guidance.py) |
| Eqs. 6–7 — adversarial noise selection (`J`, `omega`, `Norm`) | [`dpms_ant/adversarial_noise.py`](dpms_ant/adversarial_noise.py) |
| Section 4.3 / Eq. 8 — adaptor module, `x^l = theta^l(x^{l-1}) + psi^l(x^{l-1})` | [`dpms_ant/adaptor.py`](dpms_ant/adaptor.py) |
| Algorithm 1 — the full training loop | [`dpms_ant/ant_trainer.py`](dpms_ant/ant_trainer.py) |
| Section 5.2 — pre-trained DDPM / classifier backbones | [`dpms_ant/backbones.py`](dpms_ant/backbones.py), [`dpms_ant/third_party/guided_diffusion`](dpms_ant/third_party/guided_diffusion) |
| Section 5.2 — LDM backbone | [`dpms_ant/ldm_backend.py`](dpms_ant/ldm_backend.py) |
| Section 5.2/5.5 — binary domain classifier | [`dpms_ant/classifier.py`](dpms_ant/classifier.py) |
| Section 5.2 — datasets | [`dpms_ant/data.py`](dpms_ant/data.py), [`scripts/prepare_datasets.py`](scripts/prepare_datasets.py) |
| Section 5.2 — Intra-LPIPS and FID | [`dpms_ant/metrics.py`](dpms_ant/metrics.py) |
| Section 5.1 / Figure 2 — toy study | [`dpms_ant/toy_experiment.py`](dpms_ant/toy_experiment.py) |
| Section 5.2 + supplementary — per-task hyper-parameters | [`dpms_ant/configs.py`](dpms_ant/configs.py) |
| end-to-end experiment driver | [`dpms_ant/image_experiments.py`](dpms_ant/image_experiments.py) |

### The three method components

**Similarity-guided training (Eq. 5).** A frozen binary classifier
`p_phi(y | x_t)`, trained on *noised* images, provides
`grad_{x_t} log p_phi(y = T | x_t)`, which is subtracted from the noise target
(the sign is exactly the paper's):

```
min_theta E || eps_t - eps_theta(x_t, t) - sigma_hat_t^2 * gamma * grad log p_phi(y=T|x_t) ||^2
sigma_hat_t = (1 - alpha_bar_{t-1}) * sqrt(alpha_t / (1 - alpha_bar_t))
```

The sign is consistent with classifier guidance at sampling time: at
convergence `eps_theta = eps - sigma_hat^2 gamma grad log p_T`, and the reverse
step subtracts `eps_theta`, so the sample moves *towards* the target domain.

**Adversarial noise selection (Eqs. 6–7).** Instead of `eps ~ N(0, I)` the
inner maximisation is solved with `J = 10` gradient-ascent steps of size
`omega = 0.02`, each followed by `Norm(.)` (per-sample zero mean / unit
standard deviation for images, which is what keeps the perturbation a valid
diffusion noise).

**Adaptor (Section 4.3).** `down-pool -> norm + 3x3 conv -> 4-head attention ->
MLP to a bottleneck of 8/16 -> x4 up-sample -> norm -> 3x3 conv`, all
zero-initialised, injected into every U-Net layer as a parallel residual. The
pre-trained weights are frozen; only `psi` is optimised, which is what keeps
the "Parameter Rate" of Table 1 small.

---

## 2. Quick start

```bash
pip install -r requirements.txt          # torch, lpips, pytorch-fid, torchvision, diffusers
python -m pytest tests -q                # 41 unit tests, ~15 s on CPU
python scripts/smoke_test.py             # full pipeline on synthetic data, CPU, ~2 min
python scripts/run_toy.py --output-dir outputs/toy    # Section 5.1 / Figure 2
```

The image experiments need the third-party datasets and the pre-trained
checkpoints:

```bash
python scripts/prepare_datasets.py --data-root data     # shows the expected layout

# 1) domain classifier (supplementary: Adam, lr 1e-4, batch 64, 300 iterations)
python scripts/train_classifier.py --source-root data/ffhq256 \
    --target-root data/few_shot/sunglasses --output outputs/sunglasses/classifier.pt

# 2) DPMs-ANT (Algorithm 1) + Intra-LPIPS / FID
python scripts/train_ant.py --task ddpm_ffhq_sunglasses \
    --source-root data/ffhq256 --target-root data/few_shot/sunglasses \
    --classifier-checkpoint outputs/sunglasses/classifier.pt \
    --output-dir outputs

# 3) the ablation rows of Figure 4
python scripts/train_ant.py --task ddpm_ffhq_sunglasses --baseline vanilla ...
python scripts/train_ant.py --task ddpm_ffhq_sunglasses --baseline full    ...

# 4) LDM backbone
python scripts/train_ant.py --task ldm_ffhq_sunglasses --checkpoint <diffusers-ldm-dir> ...

# 5) every task of Section 5.2 in one go, printed as Tables 1-2
python scripts/run_all_tasks.py --data-root data --output-dir outputs \
    --fid-root data/fid --classifier-root outputs/classifiers

# 6) sampling / metric only
python scripts/generate.py --checkpoint checkpoints/256x256_diffusion_uncond.pt \
    --adaptor outputs/ddpm_ffhq_sunglasses/checkpoints/adaptor.pt --num-samples 64
python scripts/evaluate.py --generated outputs/ddpm_ffhq_sunglasses/samples.pt \
    --target-root data/few_shot/sunglasses --real-root data/fid/sunglasses
```

`--task` accepts every row of [`dpms_ant/configs.py`](dpms_ant/configs.py)
(`ddpm_ffhq_babies`, `ddpm_ffhq_sunglasses`, `ddpm_ffhq_raphael`,
`ddpm_lsun_haunted_houses`, `ddpm_lsun_landscape_drawings`, and the five `ldm_*`
counterparts) with the exact learning rate, `C` (bottleneck `d`), `gamma`,
`omega`, `J` and iteration count given in the supplementary material.

---

## 3. Models, data and checkpoints

* **DDPM backbone** — the released unconditional 256x256 guided-diffusion model
  `256x256_diffusion_uncond.pt` (Dhariwal & Nichol, 2021), the same model family
  used by DDPM-PA. [`dpms_ant/third_party/guided_diffusion`](dpms_ant/third_party/guided_diffusion)
  contains an unmodified, MIT-licensed copy of its model definitions so the
  checkpoint loads bit-exactly.
* **Domain classifier** — `256x256_classifier.pt` with its last layer replaced
  by a 2-way head, then fine-tuned (Adam, lr `1e-4`, batch `64`, `300`
  iterations) on *noised* source/target images, exactly as described in the
  supplementary material. Section 5.5's finding that 10 target images suffice is
  reproduced by the same recipe with `--num-shots 10`.
* **LDM backbone** — any diffusers-format LDM (`UNet2DConditionModel` +
  `AutoencoderKL`/`VQModel`). The paper's LDMs come from Rombach et al.; the
  original `.ckpt` files have to be converted to the diffusers layout first
  (the conversion script ships with `diffusers`).
* **Datasets** — FFHQ / LSUN Church as sources and the 10-shot targets
  (Babies, Sunglasses, Raphael Peale, Sketches, Amedeo Modigliani; Haunted
  houses, Landscape drawings) are third-party artefacts, so only the *paths*
  are wired up (`scripts/prepare_datasets.py` prints the expected layout and
  where each dataset comes from).

---

## 4. Results obtained in this reproduction

Everything below was run on CPU in the reproduction environment (no GPU); the
paper's image experiments need GPUs and are therefore *implemented and
smoke-tested*, not trained to completion here.

### 4.1 Unit/integration checks (all passing)

`python -m pytest tests -q` → **41 passed**, covering: the schedule identities
(`alpha_bar`, `sigma_hat`), the forward/inverse process, the exact form of the
similarity-guided loss and of the classifier-gradient term, the
adversarial-noise ascent (`Norm`, monotone ascent without normalisation,
history), adaptor zero-initialisation and shape handling, adaptor-only gradient
flow with a frozen backbone, Algorithm 1's loss decrease, sampling shapes,
Intra-LPIPS behaviour (0 for copies, larger for diverse sets), the Fréchet
distance, and the LDM backend (adaptors attached to a diffusers U-Net, latent
epsilon prediction, and pixel-space classifier guidance differentiated through
the frozen decoder).

`python scripts/smoke_test.py` runs the whole pipeline end to end (U-Net →
adaptors → classifier fine-tuning on noised images → Algorithm 1 → sampling →
Intra-LPIPS) on synthetic 32x32 data and finishes in ~2 minutes on CPU; the
domain classifier reaches 100% accuracy on the synthetic source/target split.

### 4.2 Toy study (Section 5.1, Figure 2)

`python scripts/run_toy.py --source-iterations 8000 --transfer-iterations 300`
(source `N((1,1), I)`, target `N((-1,-1), I)`, 10-shot transfer; energy distance
to the target distribution, lower is better):

| iteration | 30 | 90 | 120 | 150 | 240 | 300 |
|---|---|---|---|---|---|---|
| baseline (vanilla DDPM loss) | 3.12 | 2.15 | 1.01 | 0.97 | 0.56 | 0.57 |
| DPMs-ANT w/o AN (similarity guidance only) | 3.05 | 1.79 | 0.49 | 0.45 | 4.11 | 1.12 |
| **DPMs-ANT** (guidance + adversarial noise) | 3.08 | 1.69 | **0.44** | 0.56 | — | — |
| source model, no transfer | 3.47 | | | | | |

The target region is reached in roughly half as many iterations by the two
guided variants and fastest by the full method — the "learns the distribution
more quickly" claim of Figures 2(b)/(c) — while continuing to optimise the
10 shots afterwards overshoots, i.e. the overfitting behaviour the paper
describes in Table 7. Generated figures:
`outputs/toy/figure2a_gradient_directions.png`,
`figure2b_baseline_heatmap.png`, `figure2c_ant_heatmap.png`,
`convergence.png`, `convergence_distance.png`.

The "worse-case" noise cloud stops being isotropic: with the paper's
`omega = 0.02` (global normalisation, which the 2-D toy needs — per-sample
normalisation in two dimensions collapses every perturbation onto
`(+-1, -+1)`) the covariance becomes elliptical and its principal axis
correlates with the gradient direction in noise space (|cos| up to ≈ 0.8 for
larger `omega`).

### 4.3 Deviations from the paper's reported numbers

* **Figure 2(a) (gradient direction).** The paper's ordering — our method's
  output-layer gradient closest to the 10,000-sample reference — is *not*
  stable in our runs of the 2-D setup: typical angle errors are
  `baseline ≈ 8.8°`, `SG ≈ 8.2°`, `ANT ≈ 9.6°` (seed 0) but the ordering flips
  with the random seed and with the timestep at which the study is run. The
  paper does not specify that timestep, the network, or the guidance scale for
  the toy study, and in two dimensions the 10-shot gradient is already within
  ~10° of the reference gradient, which leaves little room for a correction.
  We therefore report the numbers as measured rather than tuning to the figure;
  `--gradient-timestep-fraction`, `--seed` and `--guidance-rms` expose the
  knobs. In higher-dimensional versions of the same toy (≥16 dims, where the
  10-shot gradient *is* noisy) the paper's ordering does appear
  (`baseline 8.5°/9.3°` vs `SG 7.6°/8.2°` and `ANT 7.3°/8.2°`).
* **Adaptor parameter rate.** With the paper's bottleneck dimensions
  (`d = 8/16`, pooling factor `c = 4` for DDPM) our adaptors hold **1.83%** of
  the 551.5M-parameter DDPM (0.72% when attaching one adaptor per top-level
  block instead of per layer) against the 1.3% reported in Table 1.
  `--adaptor-granularity {layer,block}` selects the granularity and the exact
  rate is written to `results.json`.
* **LDM checkpoints.** The LDM path is implemented on top of `diffusers`
  (`UNet2DConditionModel` + `AutoencoderKL`/`VQModel`, `scaled_linear`
  schedule, `c = 2, d = 8`); the original Rombach `.ckpt` files must be
  converted to that layout first. For LDMs the diffusion variable is a latent,
  so the *image* classifier is applied by decoding the latent and
  differentiating through the frozen decoder
  ([`dpms_ant/ldm_backend.py`](dpms_ant/ldm_backend.py)).
* **FID / Intra-LPIPS values.** The tables in the paper need ~1000-2500
  generated 256x256 images per task; those runs need a GPU (the paper reports
  3 GPU-hours per model) and are therefore not reproduced here. The metrics
  themselves are implemented with the official `lpips` and `pytorch-fid`
  packages (`dpms_ant/metrics.py`) and are unit-tested.
* Appendix-only experiments (sensitivity analysis, user study, GPU-memory
  table) are out of scope; `configs.sensitivity_configs()` is provided anyway
  because the same trainer can produce them.

---

## 5. Repository layout

```
dpms_ant/
  schedules.py          forward process, alpha/alpha_bar, sigma_hat
  sampling.py           DDPM/DDIM reverse steps, sampling loop
  guidance.py           Eq. 5 loss and the classifier-gradient term
  adversarial_noise.py  Eqs. 6-7 (J-step ascent + Norm)
  adaptor.py            adaptor module + injection into a U-Net
  ant_trainer.py        Algorithm 1 (+ the two ablation baselines)
  backbones.py          guided-diffusion DDPM / classifier checkpoints
  classifier.py         binary domain classifier training (Section 5.5)
  ldm_backend.py        LDM (diffusers) latent backend
  data.py               few-shot datasets, toy Gaussians
  metrics.py            LPIPS, Intra-LPIPS, FID
  toy_experiment.py     Section 5.1 / Figure 2
  image_experiments.py  end-to-end driver for the image tasks
  configs.py            the per-task hyper-parameters of the paper
  third_party/guided_diffusion/  vendored, MIT-licensed model definitions
scripts/                run_toy, train_classifier, train_ant, run_all_tasks,
                        generate, evaluate, prepare_datasets, smoke_test
tests/                  41 unit/integration tests
```

## 6. Environment notes

* CPU-only, `torch >= 2.0`. `--device auto` selects CUDA when present, else
  CPU; Apple's MPS must be requested explicitly.
* `lpips`, `pytorch-fid` and `torchvision` are optional but recommended: the
  metrics then match the reference implementations (without them,
  `dpms_ant.metrics` falls back to re-implementations and warns).
* All artefacts (`.pt`, datasets, generated images, logs) are written under
  `outputs/`, `checkpoints/` and `data/`, which are git-ignored: the repository
  only contains source code.
