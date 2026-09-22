# Baselines for DPMs-ANT

This directory contains the **evaluation scaffolding** for the comparison baselines used in
*"Adapting Pretrained Diffusion Models for Few-Shot Image Generation"* (DPMs-ANT).

We deliberately **do not re-implement** the baseline methods. Instead we provide:

1. **Command templates** for each official baseline codebase (`BASELINES` in `eval_baselines.py`).
2. **Drivers** that render and execute those templates (`eval_baselines.py`, `run_ddpm_pa.sh`).
3. **Evaluation hooks** that score generated images with the *same* metric implementations used
   for DPMs-ANT (`dpm_ant/evaluation/intra_lpips.py`, `dpm_ant/evaluation/fid.py`,
   `dpm_ant/evaluation/evaluate.py`), so the numbers are directly comparable to Tables 1–2, 4, 9.

---

## 1. Files

| File | Purpose |
| --- | --- |
| `eval_baselines.py` | Registry of baselines, command rendering/execution, metrics, paper-table diffing, CLI. |
| `run_ddpm_pa.sh` | Bash driver that runs DDPM-PA (and, with `--method`, any registered baseline) per task. |
| `README.md` | This document. |

---

## 2. Baselines covered

| Key | Method | Venue | Codebase | Kind |
| --- | --- | --- | --- | --- |
| `ddpm_pa` | DDPM-PA (DiffusionCLIP-style prior/appearance adaptation) | Zhu et al. 2022 | `external/DDPM-PA` | diffusion |
| `tgan` | TGAN + EWC | Wang et al. 2018 | `external/TGAN` | GAN |
| `tgan_ada` | TGAN + ADA | Karras et al. 2020 (+ Wang et al. 2018) | `external/TGAN-ADA` | GAN |
| `ewc` | EWC (elastic weight consolidation transfer) | Li et al. 2020 | `external/EWC` | GAN |
| `cdc` | CDC (cross-domain correspondence) | Ojha et al. 2021 | `external/CDC` | GAN |
| `dcl` | DCL (diverse generative few-shot) | Zhao et al. 2022 | `external/DCL` | GAN |

Each baseline expects to be checked out under `external/` (configurable through
`--external-root` / `EXTERNAL_ROOT`). If a repository is missing, the driver still renders the
command (useful with `--dry-run`) but the actual run fails gracefully with a warning.

```bash
# list the registry + availability flags
python baselines/eval_baselines.py --list

# print the paper reference tables (Table 1/2/4/9 numbers)
python baselines/eval_baselines.py --paper
```

---

## 3. Directory layout

Outputs mirror the DPMs-ANT pipeline so that `evaluate_from_dirs` can score them unchanged:

```
outputs/baselines/
├── logs/
│   └── run_ddpm_pa.log
├── ddpm_pa/
│   └── ffhq_sunglasses_ddpm/
│       ├── samples/                 # 1,000 generated images (Intra-LPIPS)
│       ├── samples_fid/             # generated images for FID (2,500)
│       └── report.json              # rendered commands + metrics + paper diff
├── cdc/
└── ...
```

---

## 4. Usage

### 4.1 DDPM-PA (the main diffusion baseline)

```bash
# full run: train + sample + evaluate on the five headline tasks
bash baselines/run_ddpm_pa.sh

# evaluate already-generated images only (no training/sampling)
EVAL_ONLY=1 bash baselines/run_ddpm_pa.sh

# smoke test the plumbing on a single task with tiny sample counts
SMOKE=1 bash baselines/run_ddpm_pa.sh

# dry run: print the exact commands that would be executed
DRY_RUN=1 bash baselines/run_ddpm_pa.sh
```

Environment knobs (all optional, defaults in parentheses):

| Variable | Meaning |
| --- | --- |
| `PYTHON_BIN` | Python interpreter (`python`) |
| `CONFIG` / `PER_TASK_CONFIG` / `CLASSIFIER_CONFIG` | YAML configs (`configs/*.yaml`) |
| `METHOD` | Baseline key (`ddpm_pa`) |
| `TASKS` | Space-separated tasks (`ffhq_babies ffhq_sunglasses ffhq_raphael church_haunted_houses church_landscape_drawings`) |
| `ALLOW_LDM` | Include `*_ldm` tasks (`0`) — DDPM-PA is pixel-space, so LDM tasks are skipped by default |
| `EXTERNAL_ROOT` | Where baseline repos live (`external`) |
| `OUT_ROOT` | Output root (`outputs/baselines`) |
| `SOURCE_DIR` / `TARGET_DIR` / `FID_TARGET_DIR` | Optional explicit dataset paths |
| `SHOTS` | Few-shot count (`10`) |
| `IMAGE_SIZE` | Resolution (`256`) |
| `NUM_SAMPLES` | Images for Intra-LPIPS (`1000`) |
| `FID_NUM_SAMPLES` | Images for FID (`2500`) |
| `FID_BACKEND` | `clean-fid` / `pytorch-fid` / `torch` |
| `DEVICE` / `GPU` / `SEED` | Runtime selection |
| `SKIP_TRAIN` / `SKIP_SAMPLE` / `SKIP_EVAL` | Skip individual stages |
| `STRICT` | Abort on first task failure |
| `SMOKE` / `DRY_RUN` / `VERBOSE` | Modes |

### 4.2 Any registered baseline

```bash
python baselines/eval_baselines.py \
  --method cdc --task ffhq_sunglasses --backbone ddpm \
  --external-root external --out-root outputs/baselines \
  --shots 10 --image-size 256 --num-samples 1000 --fid-num-samples 2500 \
  --fid-backend clean-fid --seed 0 --gpu 0 \
  --config configs/default.yaml --per-task-config configs/per_task.yaml \
  --classifier-config configs/classifier.yaml \
  --report outputs/baselines/cdc/ffhq_sunglasses_ddpm/report.json
```

Useful flags: `--list`, `--paper`, `--eval-only`, `--skip-train`, `--skip-sample`,
`--skip-eval`, `--dry-run`, `--verbose`, `--no-intra-lpips`, `--no-fid`.

---

## 5. Command templates

`eval_baselines.py` fills these placeholders:

`{repo} {python} {source_dir} {target_dir} {fid_target_dir} {out_dir} {samples_dir}
{shots} {seed} {image_size} {gpu} {num_samples} {batch_size}`

Example (DDPM-PA, abbreviated from the registry):

```bash
# train
{python} {repo}/main.py --train --source {source_dir} --target {target_dir} \
    --shots {shots} --image_size {image_size} --seed {seed} --gpu {gpu} \
    --out_dir {out_dir}
# sample
{python} {repo}/main.py --sample --out_dir {out_dir} --samples_dir {samples_dir} \
    --num_samples {num_samples} --batch_size {batch_size} --seed {seed} --gpu {gpu}
```

The exact CLI of each upstream repo changes over time — edit the `train_cmd` / `sample_cmd`
entries in the `BASELINES` registry to match your checkout. All other logic (metrics,
aggregation, reporting) is upstream-independent.

---

## 6. Metric pipeline

Both metrics reuse the ANT implementations:

* **Intra-LPIPS** (`dpm_ant.evaluation.intra_lpips.compute_intra_lpips`, §5.2): 1,000 generated
  images are assigned to their nearest *training* image by LPIPS distance; pairwise LPIPS within
  each cluster is averaged and then averaged across clusters. **Higher is better**; 0 means the
  model reproduces training samples exactly.
* **FID** (`dpm_ant.evaluation.fid.compute_fid`, §5.2): computed against the larger target sets
  (Babies 2.7k, Sunglasses 2.5k) following DDPM-PA. **Lower is better**. 10-shot FID is disabled
  by default (`compute_10shot=False`) because the paper notes it is unstable.

Reference size table (`TARGET_FID_SIZES`):

| Target | Reference images |
| --- | --- |
| `babies` | 2700 |
| `sunglasses` | 2500 |

---

## 7. Paper reference numbers

These constants are embedded in `eval_baselines.py` (`PAPER_INTRA_LPIPS`, `PAPER_FID`,
`PAPER_USER_STUDY`) and in `dpm_ant/evaluation/evaluate.py`; `compare_to_paper()` diffs the
measured values against them.

### Table 1 — Intra-LPIPS (↑), FFHQ → target

| Method | Babies | Sunglasses | Raphael | — |
| --- | --- | --- | --- | --- |
| TGAN | 0.528 | 0.506 | 0.537 | |
| TGAN+ADA | 0.609 | 0.546 | 0.583 | |
| EWC | 0.360 | 0.334 | 0.311 | |
| CDC | 0.638 | 0.582 | 0.609 | |
| DCL | 0.646 | 0.596 | 0.609 | |
| DDPM-PA | 0.656 | 0.606 | 0.619 | |
| **DDPM-ANT** | **0.661** | **0.613** | **0.632** | |
| **LDM-ANT** | **0.685** | **0.618** | **0.639** | |

### Table 1 — Intra-LPIPS (↑), LSUN Church → target

| Method | Haunted Houses | Landscape drawings |
| --- | --- | --- |
| TGAN | 0.621 | 0.660 |
| TGAN+ADA | 0.629 | 0.679 |
| EWC | 0.412 | 0.410 |
| CDC | 0.652 | — |
| DCL | — | — |
| DDPM-PA | 0.694 | 0.706 |
| **DDPM-ANT** | **0.700** | **0.723** |
| **LDM-ANT** | **0.709** | **0.738** |

### Table 2 — FID (↓), FFHQ → target

| Method | Babies | Sunglasses |
| --- | --- | --- |
| DDPM-PA | 66.22 | 23.47 |
| **DDPM-ANT** | **46.70** | **20.06** |

### Table 4 — Additional targets (Intra-LPIPS ↑)

| Method | Sketches | Amedeo's paintings |
| --- | --- | --- |
| DDPM-ANT | 0.544 ± 0.025 | 0.620 ± 0.021 |

### Table 9 — User study (§B.4): average preference for ANT over DDPM-PA ≈ **73.35 %**.

---

## 8. `report.json` schema

```jsonc
{
  "method": "ddpm_pa",
  "task": "ffhq_sunglasses",
  "backbone": "ddpm",
  "run": {
    "train": {"returncode": 0, "elapsed_sec": 15120.0, "output_tail": "..."},
    "sample": {"returncode": 0, "elapsed_sec": 1200.0}
  },
  "metrics": {
    "intra_lpips": 0.606,
    "intra_lpips_std": 0.020,
    "fid": 23.47,
    "num_generated": 1000,
    "num_reference": 2500,
    "backend": "clean-fid"
  },
  "paper": {
    "intra_lpips": 0.606,
    "fid": 23.47,
    "delta_intra_lpips": 0.000,
    "delta_fid": 0.00,
    "within_tolerance": true
  }
}
```

---

## 9. Efficiency context (§5.3, Table 8)

The paper's contrast that these baselines are meant to highlight:

| | Fine-tuned params | Training iterations | Wall-clock |
| --- | --- | --- | --- |
| Full fine-tuning / DDPM-PA style | 100 % | ~5,000 | ~4.2 GPU hours |
| **DDPM-ANT (adaptors only)** | **1.3 %** | **~300** | **~3 GPU hours** |
| **LDM-ANT (adaptors only)** | **1.6 %** | **~300** | **~3 GPU hours** |

Adaptor-only training fits in ~6 GB GPU memory vs ~17 GB for full fine-tuning.

---

## 10. Troubleshooting

* **Missing repository** — check out the baseline under `external/<name>` (or set
  `EXTERNAL_ROOT`), then re-run with `--skip-train`/`--skip-sample` as needed.
  `python baselines/eval_baselines.py --list` shows which repos were found.
* **Missing dataset directories** — set `SOURCE_DIR` / `TARGET_DIR` / `FID_TARGET_DIR`
  explicitly, or prepare them with `python main.py prepare_data --task <task>`.
* **No `clean-fid`/`lpips` installed** — the metric modules fall back to `pytorch-fid` and a
  pure-torch LPIPS approximation, so evaluation still completes (with a warning).
* **Numbers not matching the paper** — ensure identical preprocessing (aspect-preserving resize +
  center crop to 256×256, `[-1, 1]` normalization), the same 10-shot list, and the same
  reference sets (2,500/2,700 images) before comparing.
