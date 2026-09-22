# DPMs-ANT — Adapting Pretrained Diffusion Models for Few-Shot Image Generation

Reproduction of **“Adapting Pretrained Diffusion Models for Few-Shot Image Generation”**
(DPMs-ANT). The method adapts a *frozen* pretrained diffusion model (DDPM 256×256 or
Latent Diffusion Model 64×64) to a new target domain using only **10 target images** by

1. inserting **zero-initialized adaptors** ψ into the U-Net “shift” module
   (so the adapted network is bit-identical to the pretrained one before training), and
2. fine-tuning **only ψ** with a **similarity-guided** loss (Eq. 5) evaluated on
   **adversarial noise** (Eq. 7, Algorithm 1).

The whole pipeline lives in the `dpm_ant` package; `main.py` is the single dispatcher
for classifier training, ANT adaptation, sampling, evaluation, data prep, ablations and
the toy experiment.

---

## 1. Method at a glance (paper → code map)

| Paper element | Where it lives |
|---|---|
| §3 Preliminary, forward process q(x_t\|x_0), Eq. (2) DDPM loss | `dpm_ant/diffusion/schedule.py`, `dpm_ant/diffusion/gaussian_diffusion.py` |
| §3 Eq. (2)/(3) reverse process, DDPM/DDIM | `dpm_ant/sampling/sampler.py` |
| §4.1 Eq. (4) classifier-guided reverse, Eq. (5) similarity-guided loss | `dpm_ant/diffusion/gaussian_diffusion.py`, `dpm_ant/training/sg_loss.py` |
| §4.1 frozen classifier p_φ(y\|x_t), ∇_{x_t} log p_φ(y=T\|x_t) | `dpm_ant/models/classifier.py`, `dpm_ant/training/classifier_train.py` |
| §4.2 Eq. (7) adversarial-noise inner maximization, J=10, ω=0.02 | `dpm_ant/training/adv_noise.py` |
| §4.3 Eq. (8) / Algorithm 1 adaptor-only ANT training | `dpm_ant/training/ant_trainer.py` |
| §4.3 + §5.2 zero-init adaptor ψ^l(x)=f(xW_down)W_up (c=4/d=8 DDPM, c=2/d=8 LDM) | `dpm_ant/models/adaptor.py` |
| §3, §5.2 frozen DDPM U-Net / LDM U-Net + autoencoder | `dpm_ant/models/unet_loader.py`, `dpm_ant/models/ldm_loader.py` |
| §5.1, Figure 2 toy 2-D Gaussians | `dpm_ant/toy/toy_2d.py`, `dpm_ant/toy/toy_plots.py` |
| §5.2 Intra-LPIPS, FID | `dpm_ant/evaluation/intra_lpips.py`, `dpm_ant/evaluation/fid.py` |
| §5.3/Table 1/Table 8 param rate, GPU memory, wall-clock | `dpm_ant/evaluation/metrics.py` |
| §5.4 ablation (Figure 4), §5.5/Table 3 classifier pool, App. B.3 Tables 5–7 | `scripts/run_ablation.py` |
| §5.2 baselines (DDPM-PA, TGAN, TGAN+ADA, EWC, CDC, DCL) | `baselines/eval_baselines.py`, `baselines/run_ddpm_pa.sh` |

---

## 2. Environment

```bash
# Python 3.8–3.10, PyTorch >= 1.13 (2.x fine) built for your CUDA (>= 11.3)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Backbone codebases are **vendored**, not pip-installed (fork/checkpoint compatibility):

```bash
mkdir -p external
git clone https://github.com/openai/guided-diffusion external/guided-diffusion   # DDPM 256x256
git clone https://github.com/CompVis/latent-diffusion external/latent-diffusion   # LDM 64x64 + KL-AE
```

The loaders (`unet_loader.py`, `ldm_loader.py`) ship **self-contained replicas** with
identical parameter names, so the pipeline also runs without the vendored repos; the
replicas take over whenever `guided_diffusion.*` / `ldm.*` cannot be imported.

### Checkpoints (downloaded, not committed)

| File | Source |
|---|---|
| `checkpoints/256x256_diffusion_uncond.pt` | OpenAI guided-diffusion (`guided-diffusion/256x256_diffusion_uncond.pt`) |
| `checkpoints/256x256_classifier.pt` | `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt` |
| `checkpoints/64x64_classifier.pt` | `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt` |
| LDM U-Net weights + KL autoencoder | Rombach et al. 2022 (CompVis latent-diffusion releases) |

Paths are read from `models.ddpm.*` / `models.ldm.*` in `configs/default.yaml`
(or CLIs: `--diffusion-checkpoint`, `--classifier-checkpoint`,
`--autoencoder-checkpoint`). Missing checkpoints never crash the pipeline: the
loaders warn and fall back to (random) initialization so smoke tests still run.

---

## 3. Data layout

```bash
python main.py prepare_data --print-layout     # show the tree
python main.py prepare_data --prepare-all      # create dirs, resize/crop to 256x256, sample 10-shot
python main.py prepare_data --latents          # optional: cache 64x64 LDM latents (f=4, scale 0.18215)
python main.py prepare_data --verify           # verify every task's files
```

Expected layout (override root with `DPM_ANT_DATA_ROOT` or `--root`):

```
data/
  source/
    ffhq/                      # FFHQ source images
    lsun_church/               # LSUN Church source images
  targets/
    babies/ sunglasses/ raphael/ sketches/ amedeo/ haunted_houses/ landscape_drawings/   # 10-shot each
  fid_targets/
    babies/                    # 2,700 images (Table 2 FID reference)
    sunglasses/                # 2,500 images (Table 2 FID reference)
  raw/                         # downloads / archives
  latents/                     # optional cached LDM latents (+manifest.json)
```

All images are resized preserving aspect ratio and center-cropped to **256×256** and
stored as `[-1, 1]` tensors; LDM uses the same 256×256 images, then the frozen
autoencoder encodes them to **64×64×4** latents (scale factor `0.18215`).
Directory names are resolved from `data.*` keys in the configs, never hard-coded, and
aliases such as `raphael_paintings`, `Amedeo's paintings` are normalized
(`resolve_split`). The URLs for FFHQ / LSUN / the 10-shot target sets are **not**
provided by the paper; `dpm_ant/data/prepare_data.py` exposes a URL registry left
`None` by default — supply `--url` or `data.urls.<name>` (see §9 ambiguities).

---

## 4. Quick start

```bash
# 0. Toy 2-D Gaussian sanity check (Figure 2) — no checkpoints or datasets needed
bash scripts/run_toy.sh                       # SMOKE=1 for a 20-iteration smoke test

# 1. Fine-tune the binary source/target classifier p_phi  (300 iters, Adam, lr 1e-4, bs 64)
python main.py train_classifier --task ffhq_sunglasses --backbone ddpm

# 2. Adapt the frozen backbone (adaptor-only, Algorithm 1 / Eq. 8)
python main.py train_ant --task ffhq_sunglasses --backbone ddpm \
    --gamma 5.0 --omega 0.02 --J 10 --iterations 300 --norm per_sample

# 3. Generate 1,000 images (DDIM eta=0, 100 steps) and evaluate
python main.py sample   --task ffhq_sunglasses --backbone ddpm --num-samples 1000
python main.py evaluate --task ffhq_sunglasses --backbone ddpm

# 4. Or the whole pipeline (classifier -> ANT -> sample -> evaluate) for every task
python main.py all --tasks ffhq_babies ffhq_sunglasses --backbones ddpm ldm
bash scripts/run_eval.sh                      # equivalent bash driver (logs + JSON reports)
```

Everything is driven by three YAML files merged with CLI overrides
(priority: **CLI > per-task entry > block > paper default**):

* `configs/default.yaml` — global paper hyperparameters and the task registry
* `configs/per_task.yaml` — per-task overrides (lr, C/d, ω, J, γ, iterations) + sweeps
* `configs/classifier.yaml` — classifier fine-tuning settings + `classifier_ablation`

---

## 5. Hyperparameters (paper-specified) and resolved ambiguities

Defaults come from §5.2 / the addendum and are centralized in `configs/default.yaml`
(and mirrored as `ANT_DEFAULTS` in `scripts/train_ant.py`, `ANTConfig` in
`dpm_ant/training/ant_trainer.py`):

| Symbol | Meaning | Default |
|---|---|---|
| T | diffusion steps | 1000 (linear β schedule; cosine opt-in) |
| γ | similarity-guided weight (Eq. 5/8) | 5 |
| ω | adversarial-noise step (Eq. 7) | 0.02 |
| J | inner ascent steps (Eq. 7) | 10 |
| batch size | target batch for ANT | 40 |
| lr | adaptor Adam lr | 5e-5 (DDPM) / 1e-5 (LDM) |
| iterations | outer ANT iterations | 300 (per-task 160–500) |
| c, d | adaptor bottleneck compression / width | c=4, d=8 (DDPM); c=2, d=8 (LDM) |
| classifier | Adam, lr, bs, iters, t | 1e-4, 64, 300, t~Uniform({1..T}) |
| sampling | DDIM η=0, steps | 100 (DDPM η=1, T=1000 optional) |

Ambiguities not stated in the paper and the defaults chosen here (all overridable):

* **Noise schedule** — T=1000 linear β (Ho et al.); `schedule: cosine` is configurable.
* **Adaptor optimizer** — Adam with the reported learning rates (DDPM 5e-5, LDM 1e-5).
* **Norm(·) in Eq. (7)** — per-sample standardization over all elements
  (`ant.norm: per_sample`); `per_channel` is also implemented.
* **Bottleneck c vs. addendum C** — §5.2 (c=4 DDPM, c=2 LDM) is the default;
  the addendum’s per-task `C` (8/16) is applied as a down/up-projection compression
  override via `tasks.<name>.C`.
* **Classifier gradient** — ∇_{x_t} log p_φ(y=T|x_t) is **detached**
  (`classifier.detach_grad: true`); a `create_graph` flag exists for experiments.
* **Sampling schedule** — DDIM η=0 with 100 steps for evaluation; DDPM 1000 steps optional.
* **Full-model ablation FID** — the paper quotes 20.66 (Figure 4) and 20.06 (Table 2/3)
  for the same setting; `scripts/run_ablation.py` aggregates several seeds and reports
  mean ± std instead of picking one number.

---

## 6. Expected results (validation targets)

The paper’s reported numbers are embedded in the code as constants so that runs
auto-diff themselves against the paper (`paper_reference` in `configs/default.yaml`,
`PAPER_*` in `dpm_ant/evaluation/{evaluate,metrics,fid}.py`, `PAPER_*` in
`baselines/eval_baselines.py`, `ABLATION_VARIANTS` in `scripts/run_ablation.py`).

**Table 1 — Intra-LPIPS (↑), 10-shot, DDPM-ANT / LDM-ANT / DDPM-PA**

| source → target | metric | notes |
|---|---|---|
| LSUN Church → Landscape drawings | **0.723** (DDPM-ANT), **0.738** (LDM-ANT) vs 0.706 (DDPM-PA) | ANT best on most tasks |
| parameter rate | **1.3 %** (DDPM-ANT), **1.6 %** (LDM-ANT) | fine-tuned / total params |

**Table 2 — FID (↓), 10-shot**

| task | FID |
|---|---|
| FFHQ → Babies | **46.70** |
| FFHQ → Sunglasses | **20.06** |

**Table 3 — classifier pool ablation (FFHQ → Sunglasses, §5.5)**

| classifier trained on | Intra-LPIPS (↑) | FID (↓) |
|---|---|---|
| 10 images | 0.613 ± 0.023 | 20.06 |
| 100 images | 0.637 ± 0.013 | 22.84 |

**Table 4 — appendix tasks** — FFHQ → Sketches Intra-LPIPS `0.544 ± 0.025`;
FFHQ → Amedeo’s paintings `0.620 ± 0.021`.

**Figure 4 — ablation (FFHQ → Sunglasses, 300 iterations, FID ↓)**

| variant | FID |
|---|---|
| direct full-model fine-tuning | 41.88 |
| adaptor-only | 38.65 |
| DPMs-ANT w/o AN (similarity-guided only) | 26.41 |
| full DPMs-ANT | 20.66 (≈ 20.06 in Table 2/3) |

Qualitative progression: no sunglasses → sunglasses → richer high-frequency details.

**Tables 5–7 — sensitivity (FFHQ → Sunglasses)** — best at **γ = 5**, **ω = 0.02**,
**300 iterations** (FID 18.13). Grids: γ ∈ {1,3,5,7,9}, ω ∈ {0.01…0.05},
iterations ∈ {0,100,200,300,400}.

**Table 8 / §5.3 — efficiency** — adaptor-only training fits in ≈ 6 GB vs ≈ 17 GB for
full fine-tuning; ≈ 300 iterations ≈ **3 GPU hours** vs baseline
(5,000 iterations) ≈ **4.2 GPU hours**.

**Table 9 (App. B.4) — user study** — anonymous A/B vs DDPM-PA, 60 participants,
ANT preferred **73.35 %**.

**Figure 2 — toy 2-D Gaussians (§5.1)** — source N((1,1), I) → target N((−1,−1), I).
Full DPMs-ANT’s gradient direction is the closest to the 10,000-sample reference
(≈ 45° south-west); the adversarial noise cloud turns from a circle into an ellipse
whose principal axis follows the model-parameter gradient; the (timestep × sampled
value) heat-maps show a brighter central highlight for ANT and roughly parallel
sampling trajectories versus the baseline.

---

## 7. Ablations, sensitivity and the toy experiment

```bash
# Figure 4 + Tables 5-7 + Table 3, all in one driver (writes JSON + CSV + TXT reports)
python scripts/run_ablation.py --mode all --task ffhq_sunglasses --seeds 0 1 2

python scripts/run_ablation.py --mode ablation    --task ffhq_sunglasses
python scripts/run_ablation.py --mode sensitivity --task ffhq_sunglasses \
       --gammas 1 3 5 7 9 --omegas 0.01 0.02 0.03 0.04 0.05 \
       --iterations 0 100 200 300 400
python scripts/run_ablation.py --mode classifier  --task ffhq_sunglasses --pools 10 100

# Toy Figure 2 (SMOKE=1 for a fast smoke test)
bash scripts/run_toy.sh
python -m dpm_ant.toy.toy_2d --out-dir outputs/toy --seed 0        # experiment -> JSON/PT
python -m dpm_ant.toy.toy_plots --results outputs/toy/toy_results.json --out-dir outputs/toy
```

Variants are implemented as config flags rather than separate code paths:
`full_finetune` (all params, `only_adaptor: false`, `freeze_backbone: false`),
`adaptor_only`, `ant_wo_an` (`use_adv_noise: false` — a.k.a. DPMs-ANT w/o AN),
`full_ant`. `--no-adv-noise` / `--full-finetune` / `--variant` are exposed on
`scripts/train_ant.py` too.

---

## 8. Baselines

Baselines are **not re-implemented**; the scaffolding renders the official
train/sample commands and then scores the generated images with the *same*
Intra-LPIPS/FID code for apples-to-apples numbers.

```bash
bash baselines/run_ddpm_pa.sh                 # DDPM-PA (Zhu et al. 2022)
python baselines/eval_baselines.py --list     # registry: ddpm_pa tgan tgan_ada ewc cdc dcl
python baselines/eval_baselines.py --paper    # print the paper's reference tables
# Repositories are expected under external/{DDPM-PA,TGAN,TGAN-ADA,EWC,CDC,DCL}
```

See `baselines/README.md` for the command-template placeholders, output layout
(`outputs/baselines/<method>/<task>_<backbone>/`) and the `report.json` schema.
Score pre-generated images directly with
`python scripts/sample.py --dry-run`-style flows or
`evaluate_from_dirs(generated_dir, reference_dir, ...)`.

---

## 9. Reproducing the paper, phase by phase

| Phase | What to run | Deliverable / check |
|---|---|---|
| 0 Scaffolding | `python -c "import dpm_ant"`; `python main.py list` | importable package, schedule sanity verified against hand-computed ᾱ_t, σ_t, σ̂_t |
| 1 Backbones + classifier | `python main.py train_classifier --task ffhq_sunglasses` | 2-way classifier; ρ(target) higher on target-noised images, lower on source; Table 3 trend |
| 2 Adaptor + SG loss | `python main.py train_ant --variant ant_wo_an …` | zero-init adaptor reproduces pretrained output exactly; Sunglasses FID ≈ 26.41 |
| 3 Adversarial noise + full ANT | `python main.py train_ant --variant full_ant …` | inner ascent raises the inner loss, outer loss decreases; only ψ receives gradients; FID ≈ 20 |
| 4 Sampling + evaluation | `python main.py sample` / `main.py evaluate`, `bash scripts/run_eval.sh` | Tables 1–4, Table 8, Figure 3/5-style gallery |
| 5 Toy experiment | `bash scripts/run_toy.sh` | Figure 2(a)(b)(c) |
| 6 Ablations/sensitivity/efficiency | `python scripts/run_ablation.py --mode all` | Figure 4, Tables 3, 5–8 |
| 7 Baselines + docs | `bash baselines/run_ddpm_pa.sh` | Tables 1–2, 4, 9 comparisons |

### End-to-end smoke tests

```bash
# 1) zero-init adaptor must be a no-op: adapted output == pretrained output
python -c "
import torch; from dpm_ant.models.unet_loader import build_unet_model, insert_adaptors
from dpm_ant.models.adaptor import build_adaptor_factory
m = build_unet_model().eval(); x = torch.randn(1,3,64,64); t = torch.tensor([10])
with torch.no_grad(): y0 = m(x, t)
insert_adaptors(m, build_adaptor_factory(backbone='ddpm'))
with torch.no_grad(): y1 = m(x, t)
assert torch.allclose(y0, y1, atol=1e-6), 'zero-init adaptor changed the output'
print('zero-init adaptor is a no-op ✔')"

# 2) one ANT iteration updates only adaptor parameters
python -c "
import torch; from dpm_ant.models.unet_loader import build_unet_model, insert_adaptors
from dpm_ant.models.adaptor import build_adaptor_factory, adaptor_parameters
from dpm_ant.training.ant_trainer import ANTTrainer
m = build_unet_model(); insert_adaptors(m, build_adaptor_factory(backbone='ddpm'))
tr = ANTTrainer(m, config={'iterations':1,'batch_size':2,'J':2})
before = {n: p.detach().clone() for n,p in m.named_parameters()}
tr.train(target_data=torch.randn(10,3,64,64), iterations=1, verbose=False)
assert all(torch.equal(before[n], p) for n,p in m.named_parameters() if '.adaptor.' not in n)
print('only adaptor parameters changed ✔')"

# 3) tiny end-to-end run
python main.py all --tasks ffhq_sunglasses --backbones ddpm --smoke
```

(`--smoke` collapses iterations/samples to tiny values; every stage degrades
gracefully when datasets or checkpoints are absent.)

---

## 10. Repository layout

```
dpm_ant/
  diffusion/   schedule.py  gaussian_diffusion.py        # β/α/ᾱ/σ/σ̂, Eq. (2)(3)(4)(5)
  models/      unet_loader.py ldm_loader.py adaptor.py classifier.py
  training/    classifier_train.py sg_loss.py adv_noise.py ant_trainer.py
  sampling/    sampler.py                                # DDPM/DDIM reverse process
  evaluation/  intra_lpips.py fid.py metrics.py evaluate.py
  toy/         toy_2d.py toy_plots.py                    # §5.1 / Figure 2
  data/        datasets.py prepare_data.py
configs/       default.yaml per_task.yaml classifier.yaml
scripts/       train_classifier.py train_ant.py sample.py run_eval.sh
               run_ablation.py run_toy.sh
baselines/     eval_baselines.py run_ddpm_pa.sh README.md
main.py  requirements.txt  README.md
```

Outputs (defaults, override with `logging.*` / `--out`):

```
outputs/
  classifier/classifier_<task>_<backbone>.pt      # + .json metadata
  checkpoints/<task>_<backbone>_ant.pt            # adaptor-only ψ, tiny
  samples/<task>_<backbone>/                      # 1,000 PNGs (+ optional .pt)
  reports/<task>_<backbone>.json                  # Intra-LPIPS / FID / efficiency
  logs/                                           # run_eval.sh, run_ablation.py
  toy/  toy_results.json figure2a|b|c.png toy_source_model.pt toy_ant_model.pt
```

Only adaptor parameters are saved (`adaptor_state_dict()`), which is what makes the
1.3 % / 1.6 % parameter rate and the ~300-iteration, ~3-GPU-hour adaptation possible.

---

## 11. Troubleshooting

* **`lpips` / `clean-fid` / `torchvision` missing** — evaluation degrades to a
  pure-torch LPIPS and a dependency-free InceptionV3 FID path; install them for
  paper-faithful numbers.
* **No GPU** — everything falls back to CPU; training/eval will be extremely slow,
  the toy experiment and smoke tests still complete.
* **Missing datasets/checkpoints** — loaders emit warnings and use synthetic /
  randomly-initialized stand-ins so plumbing can be verified.
* **10-shot FID** — intentionally disabled (`evaluation.fid.compute_10shot: false`)
  because the paper notes it is unstable; only the 2.5k/2.7k FID reference sets are
  used by default.
* **Reproducibility** — `seed` is honored everywhere (`torch`, `numpy`, `random`);
  sample `x_0` batches come from a seeded 10-shot loader, and per-task sampling uses
  the configured seed so repeated runs match.
* **Result variance** — the paper reports ±std over seeds; run 3+ seeds and use
  `aggregate_seeds` / `--seeds 0 1 2`, and treat the 20.06 vs 20.66 difference as the
  same setting measured across seeds.
