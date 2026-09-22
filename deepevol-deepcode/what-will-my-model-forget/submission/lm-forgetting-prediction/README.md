# What Will My Model Forget? — Forecasting Forgotten Examples in Language Model Refinement

Reproduction code for the paper *"What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement"*.

The paper studies the problem of **forecasting which upstream pretraining examples will be forgotten** when an
instruction-tuned seq2seq LM is refined to fix a *single* online error. Given a base model `f_0`, an online
mispredicted example `(x_i, y_i) ~ D_R`, and a refinement procedure that produces `f_i` from `f_0` on that example,
we want a cheap function `g` that predicts, for every upstream example `x_j ∈ D_PT_hat`, whether

```
z_ij = 1[ f_i(x_j) != y_j ]           (Sec. 2, upstream forgetting definition)
```

**without** running the LM `f_i` over `D_PT_hat` at forecast time.

Implemented methods:

| # | Method | Role | Reference |
|---|--------|------|-----------|
| 1 | **Threshold** | frequency baseline (count past forgettings of `x_j`, threshold with `gamma`) | Sec. 3.1, Eq. 1 |
| 2 | **Fixed Logit** | non-trained logit-change transfer using frozen base-LM final-layer reps | Sec. 3.2, Sec. 4.2 |
| 3 | **Trainable Logit** | partially-interpretable trainable kernel `Θ̃(x_j,x_i)=h(x_j,y_j)h(x_i,y_i)^T` | Sec. 3.2, Eq. 2–3 |
| 4 | **Representation** | black-box `σ(⟨h(x_j,y_j), h(x_i,y_i)⟩ + b_j)` with frequency prior `b_j` | Sec. 3.3, Eq. 4 |
| 4b | **Representation w/o Prior** | ablation dropping `b_j` | Sec. 5.1 (Table 1) |
| 5 | **Replay refinement** | replay forecasted-forgotten examples to cut EM drop | Sec. 4.2, Sec. 5.2 |

Plus the **continual-refinement** evaluation of Fig. 3 and the base-EM sanity numbers of Table 7.

---

## 1. Repository layout

```
lm-forgetting-prediction/
  config/
    config.yaml                 # paths, model ids, LRs, steps, replay schedule, flags
    tasks.yaml                  # 36 D_PT tasks, 8 BART0 D_R tasks, 57 MMLU tasks, ID/OOD split
  src/
    data/
      em_eval.py                # SQuAD-2.0-style exact match + normalization
      p3_loader.py              # P3 train/test loaders (PromptSource), 100 ex/task
      mmlu_loader.py            # MMLU validation loader (57 subjects)
      dataset_builders.py       # D_PT, D_PT_hat, D_R, 60/40 split, ID/OOD split
    modeling/
      base_lm.py                # load BART0_L, FLAN-T5_L, FLAN-T5_3B, FLAN-T5_small; generate/logits
      refinement.py             # refinement engine: Head / LoRA / Full FT (K steps)
      encoder_h.py              # encoding h: base LM + 2-layer MLP (token & mean-pooled)
      caches.py                 # cache f0(x_j) top-k logits, h(x_j,y_j), prior b_j
    forgetting/
      ground_truth.py           # z_ij labels; collect f0/f_i logits for pairs
      frequency_prior.py        # log-odds prior b_j
    forecasters/
      threshold.py              # frequency-threshold baseline (gamma tuned on D_R^Train)
      logit_based.py            # fixed + trainable logit-change-transfer forecaster
      representation_based.py   # sigmoid(inner product + b_j), w/o-prior ablation
      losses.py                 # margin loss (Eq. 3), BCE, replay distillation
    replay/
      refinement_replay.py      # sequential/one-shot refinement w/ scheduled replay + selectors
    eval/
      metrics.py                # F1, precision, recall, Edit Success, EM Drop Ratio, stream averages
      evaluate.py               # Tables 1-4 + Figure 3 drivers, paper reference numbers
  scripts/
    build_datasets.py           # build D_PT / D_PT_hat / D_R artefacts
    generate_ground_truth.py    # sample pairs -> f_i -> z_ij, logits
    train_forecaster.py         # train logit / representation forecasters
    forecast_forgetting.py      # run g over D_PT_hat (cache-only inference)
    replay_refinement.py        # Tables 3/4 replay experiments
    run_continual_stream.py     # Figure 3 stream evaluation
  tests/                        # unit tests for EM, metrics, priors, caches, forecasters, replay
  requirements.txt
  README.md
```

---

## 2. Environment

Python **3.10**, 1× GPU. 24 GB+ is enough for `BART0_L` / `FLAN-T5_L`; a ~80 GB GPU is recommended for
`FLAN-T5_3B` (LoRA only).

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt          # CUDA build: --index-url https://download.pytorch.org/whl/cu121
```

Model / data assets used (downloaded at runtime):

* `facebook/bart-large` + BART0 weights (`INK-USC/ReCross`, subfolder `bart0_large`)
* `google/flan-t5-large`, `google/flan-t5-3b`, `google/flan-t5-small` (backbone of `h`)
* P3 (`bigscience/P3`) via PromptSource, or local ReCross JSON
* MMLU validation split (`cais/mmlu` or the original Berkeley CSV release)

All heavy imports (`torch`, `transformers`, `peft`, `datasets`, `promptsource`) are **lazy**, so config parsing,
metrics and every `--self-test` run in a CPU-only / library-light environment.

Run the unit tests:

```bash
python -m pytest lm-forgetting-prediction/tests -q
# or
python -m unittest discover -s lm-forgetting-prediction/tests -t .
```

Almost every CLI supports `--self-test` for a fully offline smoke check, e.g.

```bash
python lm-forgetting-prediction/scripts/generate_ground_truth.py --self-test
python lm-forgetting-prediction/scripts/train_forecaster.py --self-test
python lm-forgetting-prediction/scripts/replay_refinement.py --self-test
python lm-forgetting-prediction/scripts/run_continual_stream.py --self-test
python -m lm-forgetting-prediction.src.eval.evaluate --self-test
```

---

## 3. Pipeline

Artefacts live under `artifacts/<MODEL_KEY>[/<tuning>]/`. `MODEL_KEY ∈ {BART0_L, FLAN-T5_L, FLAN-T5_3B, FLAN-T5_small}`;
`tuning ∈ {head, lora, full_ft, none}`. The recommended default tunings are `BART0_L: full_ft`,
`FLAN-T5_L / FLAN-T5_3B: lora`.

### Phase 1 — Datasets (`scripts/build_datasets.py`)

Builds `d_pt.jsonl`, `d_pt_hat.jsonl`, `d_r.jsonl`, `d_r_train.jsonl`, `d_r_test.jsonl`, `id.jsonl`, `ood.jsonl`,
`manifest.json`.

* `D_PT` = 36 P3 **train** tasks × 100 examples = 3600.
* `D_PT_hat` = `D_PT` filtered to the examples `f_0` answers correctly (SQuAD-2.0 EM).
* `D_R` = mispredicted examples: for BART0 the 8 P3 **test** tasks, for FLAN-T5 the MMLU validation split (57 tasks).
* `D_R` split 60 / 40 into `D_R^Train` / `D_R^Test`; BART0 additionally gets the ID / OOD task split.

```bash
python lm-forgetting-prediction/scripts/build_datasets.py --model-key BART0_L
python lm-forgetting-prediction/scripts/build_datasets.py --model-key FLAN-T5_L --tuning lora
# offline smoke test:
python lm-forgetting-prediction/scripts/build_datasets.py --passthrough
```

**Sanity check (Table 7, base EM on `D_PT`):** BART0_L ≈ **50.50**, FLAN-T5_L ≈ **47.47**, FLAN-T5_3B ≈ **51.31**.
`D_PT_hat` retains the correctly answered subset, and the positive (forgotten) prevalence should land in the
**1%–10%** range.

### Phase 2 — Refinement engine (`src/modeling/refinement.py`)

`f_0 → f_i` for one online example under one of three tuning setups:

| mode | K steps | LR (single) | LR (sequential) |
|------|---------|-------------|-----------------|
| `head`     | 100 | 1e-3 (BART0_L) / 1e-4 (FLAN-T5) | 1e-3 / 1e-4 |
| `lora`     | 30  | 1e-5 (BART0_L) / 1e-4 (FLAN-T5) | 1e-6 / 1e-5 |
| `full_ft`  | 30  | 1e-5 (BART0_L) / 1e-4 (FLAN-T5) | 1e-6 / 1e-5 |

LoRA config: `r=16`, `alpha=32`, `dropout=0.1`, `bias="none"`, `target_modules=['q','v']`,
`task_type=SEQ_2_SEQ_LM`. Optimizer AdamW with `betas=(0.9, 0.999)`, `weight_decay=0.0`, grad-norm clip 1.0
(document defaults — the paper does not state optimizer settings).

Smoke test: right after fixing one error the **Edit Success Rate** should be > 95%.

```bash
python -m lm-forgetting-prediction.src.modeling.refinement --model-key BART0_L --mode head --self-test
```

### Phase 3 — Ground truth + caches (`scripts/generate_ground_truth.py`, `src/modeling/caches.py`)

For every sampled pair `(x_i,y_i) ~ D_R^Train`, `(x_j,y_j) ~ D_PT_hat`: refine `f_0 → f_i`, set
`z_ij = 1[f_i(x_j) != y_j]`, and persist the four logit streams `f_0(x_i)`, `f_i(x_i)`, `f_0(x_j)`, `f_i(x_j)`
as **top-k (k=100)** caches. Writes `pairs_train.jsonl`, `pairs_test.jsonl`, `online_train.jsonl`,
`online_test.jsonl`, `frequency_prior.json`, `run_meta.json`.

The caches (`logit_cache.pt`, `representation_cache.pt`, `frequency_prior.json`) make forecast-time inference
`O(|D_PT_hat|)` **without any LM forward pass** over the upstream pool.

```bash
python lm-forgetting-prediction/scripts/generate_ground_truth.py --model-key BART0_L --tuning head
# brute-force label verification on a small subset:
python lm-forgetting-prediction/scripts/generate_ground_truth.py --model-key BART0_L --tuning head --verify 32
```

> **Label convention.** We use the Sec. 2 upstream definition `z_ij = 1[f_i(x_j) != y_j]`. Appendix F
> Algorithm 1/3 prints `z_ij = 1[f_0(x_i) != f_i(x_i)]` (online example); we treat this as a typo and keep the
> online-example quantity only as a diagnostic field on `OnlineRecord` (`f0_correct` / `fi_correct`).

### Phase 4/5 — Forecasters (`scripts/train_forecaster.py`)

* **Threshold** (Eq. 1): `g(x_j) = 1[ #past forgottings of x_j ≥ gamma ]`, `gamma` tuned to maximize F1 on `D_R^Train`.
* **Fixed Logit** (Sec. 4.2): frozen base-LM final-layer representation in place of `h` (exact when only heads are tuned).
* **Trainable Logit** (Eq. 2–3): `f̂_i(x_j) = Θ̃[f̂_i(x_i) − f̂_0(x_i)] + f̂_0(x_j)` with
  `Θ̃(x_j,x_i) = h(x_j,y_j) h(x_i,y_i)^T ∈ R^{T×T}`, trained with the margin loss
  `max(0, 1 + (−1)^{z_ij} (max_{v≠y_j} f̂_i(x_j)[v] − f̂_i(x_j)[y_j]))`. Inference: `argmax != y_j ⇒ ẑ=1`.
* **Representation** (Eq. 4): `z̃_ij = σ(⟨h(x_j,y_j), h(x_i,y_i)⟩ + b_j)`, BCE, `--no-prior` drops `b_j`.

Encoding `h` = base-LM backbone (BART0 for BART0 experiments, FLAN-T5_small for T5 experiments) + freshly
initialized 2-layer MLP. LM LR 1e-5, MLP LR 1e-4.

Training: max **100,000** steps, batch 16 = **8 positive + 8 negative** pairs, positives down-weighted with
**alpha = 0.1**.

```bash
python lm-forgetting-prediction/scripts/train_forecaster.py --model-key BART0_L --tuning head --method representation
python lm-forgetting-prediction/scripts/train_forecaster.py --model-key BART0_L --tuning head --method logit
python lm-forgetting-prediction/scripts/train_forecaster.py --model-key BART0_L --tuning head --method logit --fixed
python lm-forgetting-prediction/scripts/train_forecaster.py --model-key BART0_L --tuning head --method representation --no-prior
```

### Phase 6 — Forecast over `D_PT_hat` (cache-only)

```bash
python lm-forgetting-prediction/scripts/forecast_forgetting.py --model-key BART0_L --tuning head \
    --method representation --max-online 50
```

Writes `forecast_predictions.jsonl` + `forecast_summary.json` (prevalence, mean score, and — when ground-truth
labels are available — precision / recall / F1).

### Phase 7 — Replay refinement (Tables 3/4)

Sequentially fix the errors of `D_R^Test`; every 10 steps replay a mini-batch of 8 (`BART0_L`, `FLAN-T5_L`) or
4 every 5 steps (`FLAN-T5_3B`). Selection ∈ {Vanilla (no replay), Random, Threshold, Logit, Representation,
Ground-Truth Forget}. Replay loss = distillation against the base-PTLM outputs (KL by default).

```bash
# Table 3 (sequential)
python lm-forgetting-prediction/scripts/replay_refinement.py --model-key BART0_L --tuning full_ft \
    --method representation --sequential
# Table 4 (single error, model reset each time)
python lm-forgetting-prediction/scripts/replay_refinement.py --model-key BART0_L --tuning full_ft \
    --method representation --single-error
```

Metrics: **Edit Success Rate** `= |{corrected}| / |D_R|`, and
**EM Drop Ratio** `= (EM(D_PT, f_i) − EM(D_PT, f_0)) / EM(D_PT, f_0)` (reported as a non-negative magnitude).

### Continual refinement (Figure 3)

Build a stream of `1/8` of `D_R`, shuffle, refine continually, compute the forecast **once at stream start and
freeze it**, then track running F1 / precision / recall up to each step. Precision stays stable while recall
degrades; Representation gives the best F1.

```bash
python lm-forgetting-prediction/scripts/run_continual_stream.py --model-key BART0_L --tuning full_ft \
    --method all --replay-method vanilla
```

Writes `<method>_stream_history.jsonl`, `<method>_stream_summary.json`, `figure3.json` (+ `figure3.png` with `--plot`).

### Reporting (`src/eval/evaluate.py`)

```bash
python -c "import sys; sys.path.insert(0,'lm-forgetting-prediction'); from src.eval import evaluate; raise SystemExit(evaluate.main(sys.argv[1:]))" \
    --paper-reference
```

`evaluate.py` renders Tables 1–4, Table 7 and Figure 3 from the persisted artefacts, embeds the paper reference
values (`PAPER_TABLE1..4`, `PAPER_BASE_EM`) and prints a diff via `compare_to_paper(...)`.

---

## 4. Expected results (paper reference values)

**Table 1 — forecasting F1 on `D_R^Test`** (per LM × tuning):

| Model / tuning | Threshold | Fixed Logit | Trainable Logit | Representation | w/o Prior |
|---|---|---|---|---|---|
| BART0_L head  | 62.96 | 69.57 | 73.39 | **79.32** | < 79.32 |
| BART0_L full_ft | 63.64 | 19.54 | 57.15 | **67.19** | < 67.19 |
| FLAN-T5_L head | 53.58 | 68.37 | — | **67.81** | < 67.81 |
| FLAN-T5_L lora | 41.42 | ~12.74 | low | **48.66** | < 48.66 |
| FLAN-T5_3B head | 55.34 | — | — | **65.93** | < 65.93 |

Ordering to reproduce: Representation is highest across all setups; Fixed Logit is strong **only** under Head
(it collapses under Full FT / LoRA where the representation itself moves); Trainable Logit helps BART0 but
underperforms on FLAN-T5; Threshold is the weak baseline. “w/o Prior” must lower Representation F1 consistently.

**Table 2 — BART0, Full FT, ID vs OOD** (F1):

| Bucket | Threshold | Representation | w/o Prior |
|---|---|---|---|
| ID  | 60.45 | **75.11** | 74.19 |
| OOD | 46.24 | **50.12** | 34.85 |

> Note: The §5.1 running text quotes `49.73` for the OOD Representation number while Table 2 prints `50.12`.
> Both are recorded (`PAPER_TABLE2` vs `PAPER_TABLE2_TEXT`); report whichever your run matches.

**Table 3 — sequential replay** (Edit Success / EM Drop Ratio):

| Model | Vanilla EM Drop | Representation EM Drop |
|---|---|---|
| BART0_L full_ft  | 9.274 | 1.634 |
| FLAN-T5_L lora   | 5.463 | 0.301 |
| FLAN-T5_3B lora  | —     | 0.138 |

Representation replay cuts EM drop drastically and stays **interior** to the GT-Forget upper bound; Edit Success
remains close to Vanilla FT.

**Table 4 — single-error case:** BART0 EM Drop `8.045` (Vanilla) → `~2.19` (Representation).

**Success criteria:** relative ordering of methods plus near-matching table numbers within run-to-run variance.
If numbers deviate, first verify (i) the ground-truth `z_ij` labels (`--verify`), (ii) the caches, and (iii) the
LR / step settings in `config/config.yaml`.

---

## 5. Configuration

Everything is centralised in `config/config.yaml` (global + per-model/per-mode hyper-parameters) and
`config/tasks.yaml` (task lists: 36 D_PT tasks, 8 BART0 D_R tasks, 57 MMLU tasks, ID/OOD split). Scripts accept
`--config` and can be run per `(model, tuning)` pair.

Documented defaults for values the paper leaves unspecified:

| Item | Value | Rationale |
|------|-------|-----------|
| seed | 42 | paper does not specify |
| optimizer | AdamW (`betas=0.9/0.999`, `wd=0.0`) | paper does not specify |
| grad clip | 1.0 | stability |
| MLP width of `h` | 768 (2 layers, GELU) | “low-dimensional” unspecified; 768 chosen |
| distillation | KL vs base-PTLM outputs (temperature 1.0) | per Buzzega 2020a |
| threshold gamma grid | 1…100 | tuned on `D_R^Train` by F1 |

---

## 6. Reconciliation notes / known ambiguities

1. **`z_ij` definition** — we follow **Sec. 2** (`f_i(x_j) != y_j`, upstream example). Appendix F’s
   `1[f_0(x_i) != f_i(x_i)]` (online example) is treated as a typo; the alternated quantity is kept as a diagnostic.
2. **OOD membership of `anli`** — `anli` appears both in the BART0 `D_R` task list and in Appendix B’s OOD list;
   we place it in **OOD** (Appendix B wins for the ID/OOD partition).
3. **Table 2 OOD Representation**: text says `49.73`, table says `50.12` — both reference values are stored.
4. **`T` reconciliation** — in the token-level kernel `h` is the output-length axis `T`; the pair kernel is
   materialised as `R^{T_j × T_i}` (rectangular, per pair).
5. **Out of scope** (not implemented, per plan): MIR / OCS baselines, §5.3 / Table 5 complexity study, and the
   §5.2 “Hyperparameter Analysis”.

---

## 7. Citation

```bibtex
@inproceedings{whatwillmyforget,
  title     = {What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement},
  author    = {Anonymous},
  booktitle = {To appear},
  year      = {2024}
}
```
