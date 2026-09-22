# Stochastic Interpolants with Data-Dependent Couplings

Reproduction code for **"Stochastic Interpolants with Data-Dependent Couplings"**.

The repository implements the generalized stochastic-interpolant framework in which the base sample is
built *conditionally on the target* through a data-dependent coupling

```
rho(x0, x1) = rho1(x1) rho0(x0 | x1),      x0 = m(x1) + sigma * zeta,   zeta ~ N(0, I)
```

and the coupled velocity field is still learned by the *same simulation-free square-loss regression* used
for the standard (independent) interpolant:

```
I_t   = alpha_t x0 + beta_t x1 + gamma_t z
I_dot = alpha_dot_t x0 + beta_dot_t x1 + gamma_dot_t z
L_b   = E |b_hat_t(I_t, xi)|^2 - 2 I_dot . b_hat_t(I_t, xi)          (Eq. 22)
```

Sampling integrates the probability-flow ODE `X_dot = b_hat_t(X_t, xi)` from `t = 0` to `t = 1`
(Algorithm 2), starting from the coupled base `X_0 = m(x1) + sigma * zeta`.

---

## 1. Installation

Python **3.10+** is required.

```bash
# (recommended) fresh environment
python -m venv .venv && source .venv/bin/activate

# CUDA build of torch/torchvision first, e.g. CUDA 12.1:
pip install torch>=2.2,<3.0 torchvision>=0.17,<1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# remaining dependencies
pip install -r requirements.txt
```

`requirements.txt` pins the known-good core stack (`torch`, `torchdiffeq==0.2.3` for the Dopri solver,
`lightning-fabric` for parallelism, `datasets` for ImageNet-1k, `torch-fidelity`/`scipy` for FID-50k,
`pyyaml`, `tqdm`, `einops`, `numpy`, `Pillow`, `matplotlib`).

Everything degrades gracefully: torchdiffeq, Lightning Fabric, HuggingFace `datasets`, and the FID stack are
all optional at import time, so `--self-test` smoke checks run on a CPU-only machine.

A GPU with **>= 24 GB** memory is recommended for the reported pixel-space ImageNet runs
(U-Net base width 256 channels, batch size 32, 200,000 steps); multi-GPU training is supported through
Lightning Fabric.

---

## 2. Data

ImageNet-1k is loaded from the HuggingFace Hub:

```python
datasets.load_dataset("imagenet-1k", trust_remote_code=True)
```

The download is large (~150 GB compressed) — make sure you have disk space. Caching is controlled by the
standard `HF_HOME` / `HF_DATASETS_CACHE` environment variables, or by the `data.cache_dir` key in a config.
Images are converted to tensors in `[-1, 1]` and resized/cropped to 256 or 512; class labels are exposed for
the class-label embedding path. The low-resolution views used by super-resolution (`D(x1)`, and the
conditioning `xi = U(D(x1))`) are produced on the fly by `si/couplings/resize.py`.

For smoke tests / no-download debugging, every data entry point accepts `--synthetic`, which substitutes a
random-tensor stand-in dataset.

---

## 3. Configuration

Four task/scale configs are provided (all values follow Appendix B / the addendum; anything unspecified by
the paper is an engineering default and annotated as such in the file):

| Config | Task | Target | Conditioning |
|---|---|---|---|
| `configs/inpainting_256.yaml` | in-painting | 256x256 | 64-tile mask, p = 0.3 |
| `configs/inpainting_512.yaml` | in-painting | 512x512 | 64-tile mask, p = 0.3 |
| `configs/superres_64_256.yaml` | super-resolution | 256x256 | `U(D(x1))` from 64x64 |
| `configs/superres_256_512.yaml` | super-resolution | 512x512 | `U(D(x1))` from 256x256 |

Shared hyperparameters (Appendix B / addendum, identical for all four tasks):

```
batch_size = 32                 grad_clip  = 10000 (whole-parameter-vector norm)
steps      = 200000             weight_decay = 0
lr         = 2e-4 (Adam)        scheduler  = StepLR(gamma = 0.99, every 1000 steps)
channels = 256, dim_mults = (1,1,2,3,4), resnet_block_groups = 8
learned_sinusoidal_cond = True, learned_sinusoidal_dim = 32, random_fourier_features = False
attention_heads = 4, attention_dim_head = 64
```

Interpolant presets (`si/interpolants/coefficients.py`):

| preset | alpha_t | beta_t | gamma_t | used by |
|---|---|---|---|---|
| `linear` | `1 - t` | `t` | `sqrt(2 t (1 - t))` | (reference / SDE paths) |
| `gamma0` | `1 - t` | `t` | `0` | super-resolution (Sec. 4.2) |
| `inpainting` | `t` | `1 - t` | `0` | in-painting (Sec. 4.1) |
| `superres` | alias of `gamma0` | | | |

All presets satisfy `alpha_0 = beta_1 = 1`, `alpha_1 = beta_0 = gamma_0 = gamma_1 = 0`, and
`alpha_t^2 + beta_t^2 + gamma_t^2 > 0`.

---

## 4. Training (Algorithm 1)

```bash
# in-painting, 256x256
python train.py --config configs/inpainting_256.yaml

# super-resolution 64 -> 256
python train.py --config configs/superres_64_256.yaml

# override anything from the CLI, e.g. multi-GPU via Fabric
python train.py --config configs/inpainting_512.yaml --devices 2 --steps 200000

# CPU smoke test, no ImageNet download
python train.py --task inpainting --resolution 256 --synthetic --steps 50
```

Per gradient step (Algorithm 1):

1. draw `x1 ~ rho1` (an ImageNet image) and `zeta ~ N(0, I)`;
2. build the coupled base `x0 = m(x1) + sigma * zeta` and the conditioning `xi`
   (`xi = mask` for in-painting, `xi = U(D(x1))` for super-resolution);
3. draw `t ~ U(0, 1)` per item and form `I_t`, `I_dot` from the coefficients;
4. minimize `L_b = mean(|b_hat_t(I_t, xi)|^2 - 2 I_dot . b_hat_t(I_t, xi))`;
5. Adam step (`lr = 2e-4`), StepLR (`gamma = 0.99` / 1000 steps), grad-norm clip at `10000`, no weight decay.

Training logs, checkpoints (model / optimizer / optional EMA), and an optional `E[|I_dot_t|^2]`
transport-cost diagnostic (Proposition 3.1) are written under `training.output_dir`.

**Structural invariants enforced during training**

* in-painting: because `xi ∘ I_t = xi ∘ x1` for all `t`, the predicted velocity is masked to exactly zero on
  observed pixels (`mask_observed: true` / `coupling.mask_velocity`).
* super-resolution: the conditioning `xi = U(D(x1))` is appended to the U-Net input channels at every
  timestep (input channels = `2 * C`).

---

## 5. Sampling (Algorithm 2)

```bash
# in-painting (Fig. 3 triples + Fig. 5 probability-flow slices)
python scripts/sample_inpainting.py --config configs/inpainting_256.yaml \
    --checkpoint runs/inpainting_256/last.pt --fid --qualitative

# super-resolution (Fig. 4 / Fig. 6)
python scripts/sample_superres.py --config configs/superres_64_256.yaml \
    --checkpoint runs/superres_64_256/last.pt --fid --qualitative

# generic entry point (task chosen by the config)
python sample.py --config configs/superres_64_256.yaml --checkpoint runs/superres_64_256/last.pt
```

The sampler integrates `X_dot = b_hat_t(X_t, xi)` from `t = 0` to `t = 1` starting at
`X_0 = m(x1) + sigma * zeta` (or an explicitly observed `x0`):

* `--method dopri5` (default) uses `torchdiffeq` with adaptive tolerances (`atol = rtol = 1e-5`);
  `euler` / `midpoint` / `rk4` are available as fixed-step integrators (`--steps N`).
* For in-painting the observed pixels are re-imposed after every step
  (`xi ∘ X_t = xi ∘ x1`, `project_observed: true`), so the unmasked region is reproduced **exactly**.
* For super-resolution `xi = U(D(x1))` is re-appended at every integration step.
* `--base-mode independent` replaces the coupled base by a pure Gaussian `x0 = zeta`, reproducing the
  uncoupled-interpolant baseline.
* Optional SDE sampling (`si/samplers/sde.py`, Eqs. 11/13) is available for stochastic paths; the reported
  experiments use the deterministic ODE.

Sampling outputs (`.pt` bundles, PNG grids, triples, probability-flow slices and a `sample_summary.json`)
are written to the directory given by `--outdir`.

---

## 6. Evaluation

```bash
python evaluate.py --config configs/inpainting_256.yaml \
    --checkpoint runs/inpainting_256/last.pt --fid --qualitative

python evaluate.py --config configs/superres_64_256.yaml \
    --checkpoint runs/superres_64_256/last.pt --fid
```

`evaluate.py` orchestrates:

* **FID-50k** (`eval/fid.py`): Fréchet distance between Inception-v3 pool3 feature Gaussians of 50,000
  generated samples and the standard ImageNet reference statistics.
* **Quantitative tables** (Tables 2 and 3) plus a JSON/text report comparing measured FID to the paper values.
* **Structural checks** (Sec. 4.1): verifies that observed in-painting pixels satisfy
  `xi ∘ X_{t=1} = xi ∘ x1` and that the predicted velocity is zero on observed pixels.
* **Transport-cost comparison** (Prop. 3.1): empirically compares `E[|I_dot_t|^2]` for the coupled base versus
  the independent base and confirms the coupled value is smaller.
* **Qualitative figures** (Figs. 3–6): base/model/ground-truth triples at 256 and 512 for in-painting and
  super-resolution (64->256 and 256->512), plus temporal probability-flow slices.

### Expected results

**Table 2 — in-painting, ImageNet 256x256/512x512 (FID-50k, lower is better)**

| Method | FID-50k |
|---|---|
| Uncoupled interpolant (independent Gaussian base) | ~1.35 |
| **Dependent coupling (ours)** | **~1.13** |

**Table 3 — super-resolution 64x64 -> 256x256 (FID-50k)**

| Method | FID-50k |
|---|---|
| Improved DDPM | reported |
| SR3 | reported |
| ADM | reported |
| Cascaded Diffusion | reported |
| I^2SB | reported |
| **Dependent coupling (ours)** | **~2.13 (train) / ~2.05 (valid)** |

The SR baselines are **cited, not re-run** (`REPORTED_ONLY_BASELINES` in `evaluate.py` /
`eval/fid.py`). The in-painting baseline (~1.35) *is* reproducible locally by training with
`base_mode: independent`.

---

## 7. Tests / smoke checks

Each module ships a `_self_test()` covering its own math and shapes; they need no ImageNet download.
Representative checks:

```bash
python -m si.interpolants.interpolant      # I_0 = x0, I_1 = x1, derivatives vs finite differences
python -m si.interpolants.coefficients     # boundary conditions of alpha/beta/gamma
python -m si.couplings.mask                # 64-tile masks, empirical p ~ 0.3, channel sharing
python -m si.couplings.superres            # D/U shapes, sigma > 0, channel concatenation
python -m si.losses.velocity_loss          # zero-model loss = d sigma^2, masked/unmasked equivalence
python -m si.losses.score_loss             # L_g = d for zero net; gamma_t = 0 guard
python -m si.models.embeddings             # embedding shapes, null class
python -m si.samplers.ode                  # toy 2D inversion X_0 -> X_1, trajectory shapes
python -m si.utils.distributed             # world size / rank / collectives on 1 process
```

Two ready-made end-to-end smoke tests (tiny model, synthetic data, ODE round-trip):

```bash
python scripts/sample_inpainting.py --self-test
python scripts/sample_superres.py --self-test
python evaluate.py --self-test
python sample.py --self-test
```

---

## 8. Repository layout and paper mapping

```
couplings-si/
├── si/
│   ├── interpolants/
│   │   ├── coefficients.py    # alpha_t, beta_t, gamma_t + derivatives, presets      §3 (Def. 3.1, Eq. 1/20)
│   │   └── interpolant.py     # I_t = a x0 + b x1 + g z and I_dot                     §3 (Def. 3.1, Eq. 1/20)
│   ├── couplings/
│   │   ├── base.py            # Coupling ABC: build_x0(x1, xi) -> (x0, xi), sigma      §3.2 (Eqs. 16-19)
│   │   ├── mask.py            # 64-tile random mask, p = 0.3, per spatial location     §4.1
│   │   ├── inpainting.py      # x0 = xi*x1 + (1-xi)*zeta ; velocity zero off-mask      §4.1
│   │   ├── resize.py          # D (downsample) / U (upsample) operators                §4.2
│   │   └── superres.py        # x0 = U(D(x1)) + sigma*zeta ; xi = U(D(x1))             §4.2
│   ├── models/
│   │   ├── embeddings.py      # time + class-label (+ learned sinusoidal) embeddings   App. B
│   │   ├── unet.py            # U-Net velocity net b_hat_t(x, xi)                      App. B, §4.1/§4.2
│   │   └── score_net.py       # optional g_hat_t(x, xi) (SDE paths only)               §3.1 (Eqs. 4/6), App. A
│   ├── losses/
│   │   ├── velocity_loss.py   # L_b, Eq. 22 (+ Prop. 3.1 transport-cost estimate)      §3.4 (Eq. 22)
│   │   └── score_loss.py      # optional L_g (Eq. 7 second line)                       §3.1, App. A (Eq. 29)
│   ├── samplers/
│   │   ├── ode.py             # probability-flow ODE: Dopri / Euler / RK4              Algorithm 2
│   │   └── sde.py             # optional forward/backward SDE (Eqs. 11, 13)            §3.1 (Cor. 3.1)
│   ├── data/
│   │   ├── imagenet.py        # HuggingFace imagenet-1k loader + synthetic fallback    §4.1/§4.2
│   │   └── transforms.py      # [-1, 1] tensors, resize/crop, low-res views            §4.1/§4.2
│   └── utils/
│       ├── config.py          # YAML config loader / dataclass
│       ├── distributed.py     # Lightning Fabric + torch.distributed glue
│       └── ema.py             # optional EMA shadow weights
├── configs/                   # inpainting_256/512, superres_64_256/256_512
├── scripts/
│   ├── sample_inpainting.py   # Fig. 3/5 + FID-50k
│   └── sample_superres.py     # Fig. 4/6 + FID-50k
├── eval/
│   ├── fid.py                 # FID-50k (Inception feature Gaussians)
│   └── qualitative.py         # base/model/GT triples, probability-flow slices
├── train.py                   # Algorithm 1 training entry point
├── sample.py                  # Algorithm 2 sampling entry point
├── evaluate.py                # FID + structural checks + figures + report
└── requirements.txt
```

---

## 9. Notes on unspecified settings

Where the paper is silent, the following defaults were chosen and are marked `[engineering]` in the configs:

* **super-resolution interpolant**: `gamma_t = 0` with `alpha_t = 1 - t`, `beta_t = t` (the coupled-base
  convention used in the experiments); in-painting uses `alpha_t = t`, `beta_t = 1 - t` (explicit in Sec. 4.1).
* **super-resolution `sigma`**: small positive scalar (`0.05`) to smooth the base density off the
  low-dimensional manifold `{U(D(x1))}`; tune by validation FID if needed.
* **ODE solver / step count**: Dopri5 with `atol = rtol = 1e-5` for reported quality; the forward-Euler
  variant of Algorithm 2 is available with a configurable (large) step count.
* **Adam betas**: PyTorch defaults `(0.9, 0.999)`; **no EMA** is used (the paper does not mention it; the
  utility exists in `si/utils/ema.py` if desired).
* **class dropout** (`0.1`), `num_workers`, precision (`32-true`) and sampling batch size are engineering
  defaults.
