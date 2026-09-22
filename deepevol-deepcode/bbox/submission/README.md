# BBox-Adapter — Lightweight Adapting for Black-Box Large Language Models

Reference implementation of the paper **"Lightweight Adapting for Black-Box Large Language Models"**
(BBox-Adapter).

BBox-Adapter adapts a **frozen black-box LLM** (e.g. `gpt-3.5-turbo`, `davinci-002`,
`Mixtral-8x7B-v0.1`) by training a **small 0.1B–0.3B LM adapter `g_theta`** with a
**ranking-based Noise Contrastive Estimation (NCE) loss** derived from an
Energy-Based-Model (EBM) perspective:

* the black-box LLM is a **proposal generator** (text in → raw text out);
* the adapter `g_theta` is an **evaluator** that ranks those proposals;
* a **sentence-level beam search** uses `g_theta` for top-`k` pruning;
* an **online adaptation loop** iteratively refreshes positive samples
  (ground truth / GPT-4 human-preference feedback) and negative samples
  (previous adapted inferences), never touching logprobs, hidden states or
  gradients of the black-box model.

---

## Table of contents

1. [Repository layout](#1-repository-layout)
2. [Installation](#2-installation)
3. [Credentials / environment variables](#3-credentials--environment-variables)
4. [Quick start](#4-quick-start)
5. [Configuration knobs](#5-configuration-knobs)
6. [Method overview & code map](#6-method-overview--code-map)
7. [Reproducing the paper](#7-reproducing-the-paper)
   - [7.1 Table 2 — main results](#71-table-2--main-results-adapting-gpt-35-turbo)
   - [7.2 Table 3 — plug-and-play](#72-table-3--plug-and-play-no-retraining)
   - [7.3 Table 4 — cost](#73-table-4--cost-per-1k-questions)
   - [7.4 Table 5 — NCE vs MLM ablation](#74-table-5--nce-vs-mlm-ablation--scale-analysis)
   - [7.5 Figure 3 — beam / iteration sweeps](#75-figure-3--beam-size-and-iteration-sweeps)
   - [7.6 Table 6 — Mixtral as black box + VRAM](#76-table-6--mixtral-white-box-treated-as-black-box--vram)
   - [7.7 Table 7 — ToxiGen](#77-table-7--toxigen-toxicity-reduction)
   - [7.8 Figure 4 — case study](#78-figure-4--gsm8k-case-study)
   - [7.9 Baselines (CoT / Azure-SFT / SFT-LoRA)](#79-baselines-cot--azure-sft--sft-lora)
8. [Running without credentials (offline / mock mode)](#8-running-without-credentials-offline--mock-mode)
9. [Tests & sanity checks](#9-tests--sanity-checks)
10. [Defaults for paper-silent hyper-parameters](#10-defaults-for-paper-silent-hyper-parameters)
11. [Troubleshooting](#11-troubleshooting)
12. [Citation](#12-citation)

---

## 1. Repository layout

```
bbox_adapter/
  configs/
    default.yaml          # master defaults (Appendix H.2) — deep-merged with dataset YAMLs
    strategyqa.yaml       # 2059/229, 2-shot, Yes/No
    gsm8k.yaml            # 7473/1319, 4-shot CoT, numeric
    truthfulqa.yaml       # 717 train / 100 test, bert-base-cased, True+Info
    scienceqa.yaml        # 2000/500 (image questions excluded), 1-shot MCQ
    toxigen.yaml          # 2000/500, temperature 0.7, RoBERTa judge
  data/
    dataset_specs.py      # HuggingFace paths, split sizes, answer types, paper reference tables
    loaders.py            # load_split / load_train / load_test, Example records
    answer_extraction.py  # '####' terminator parsing, Yes/No, numeric, MCQ, TruthfulQA, toxicity
  llm/
    blackbox_client.py    # text-only Azure gpt-3.5-turbo / davinci-002, HF Mixtral, gpt-4 rater, mock
    prompts.py            # Appendix-J generator prompts + Appendix-G rater prompts
    token_cost.py         # token accounting → $/1k questions (gpt-3.5-turbo-1106 pricing)
  adapter/
    energy_model.py       # DeBERTa-v3-base/large, BERT-base-cased + scalar energy head
    regularizer.py        # alpha * E[g_theta(x,y)^2]  (L2 energy penalty, Eq. 3)
  losses/
    nce.py                # Eq.(1) softmax posterior, Eq.(2) ranking NCE, Eq.(3) gradient
    mlm.py                # masked-LM ablation (§4.5, Table 5)
  inference/
    beam_search.py        # sentence-level beam search (Eq. 4) + single-step variant
    selector.py           # highest-scoring answer selection
  training/
    buffers.py            # positive/negative stores, SEL modes, Eq.(5)/(6), outcome supervision
    online_adaptation.py  # Algorithm 1 outer T loop + AdamW theta update (Eq. 7)
  feedback/
    ai_feedback.py        # gpt-4 rater: 4 criteria, best-answer / ranked-top-5 parsing
  eval/
    metrics.py            # Acc / True+Info / Toxic% / Toxicity Prob% / Delta%
    cost.py               # training + inference cost accounting (Table 4)
    vram.py               # peak GPU memory (Table 6)
  utils/
    logging.py            # loggers, RunLogger, metric tracker, curves, checkpoint naming
    seed.py               # reproducible seeding per component
scripts/
  run_experiment.py       # main runs: Ground-Truth / AI Feedback / Combined  (Table 2)
  run_ablation.py         # NCE vs MLM, beam-size sweep, iteration sweep, alpha sweep
  run_plug_and_play.py    # transplant trained adapter onto davinci-002 / Mixtral (Table 3)
  run_toxigen.py          # ToxiGen toxicity run (Table 7)
  run_baselines_cot_sft.py# CoT prompting, Azure-SFT, SFT-LoRA baselines
tests/
  test_nce_loss.py        # Eq. (1)–(3) numerics, gradient signs, regularizer
  test_beam_search.py     # beam pruning, single-step ranking, black-box payload guard
  test_buffers.py         # SEL modes, Eq.(5)/(6), outcome supervision
  test_answer_extraction.py
requirements.txt
```

---

## 2. Installation

Paper environment (Appendix H.2): **Python 3.10.13**, AMD EPYC 7702 64-core @1.50 GHz,
**NVIDIA A100-SXM4-80GB**. BBox-Adapter itself fits on a single A100-80GB when the
black-box LLM is API-served; Mixtral-8x7B half-precision inference needs ≈90 GiB;
SFT-LoRA requires 4× A100-80GB.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

* `torch` should be the CUDA build matching your driver (e.g. `--index-url https://download.pytorch.org/whl/cu121`).
* With the **CPU wheel** everything still imports: the pipeline falls back to
  `MockLLMClient` and the offline test-suite runs end-to-end.
* Weights are downloaded on first use: `microsoft/deberta-v3-base`,
  `microsoft/deberta-v3-large`, `bert-base-cased`,
  `facebook/roberta-hate-speech-dynabench-r4-target` (ToxiGen judge),
  `mistralai/Mixtral-8x7B-v0.1`.

---

## 3. Credentials / environment variables

Secrets are read from the environment (optionally from a local `.env`); they are **never**
committed.

| Variable | Used for |
| --- | --- |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI REST calls (gpt-3.5-turbo, davinci-002, gpt-4) |
| `AZURE_OPENAI_ENDPOINT` | Azure resource endpoint, e.g. `https://<res>.openai.azure.com` |
| `AZURE_OPENAI_API_VERSION` | optional, default `2024-02-01` |
| `HF_TOKEN` | downloading gated checkpoints (Mixtral) |

Without `AZURE_OPENAI_API_KEY` the client automatically degrades to `MockLLMClient`
(`--allow-mock`, default on) so the entire pipeline is smoke-testable offline.

Deployment-name overrides (if your Azure deployments differ from the model names):

```bash
export AZURE_DEPLOYMENT_GPT35=gpt-35-turbo-deployment
export AZURE_DEPLOYMENT_DAVINCI=davinci-002
export AZURE_DEPLOYMENT_GPT4=gpt-4
```

---

## 4. Quick start

Nothing here requires Azure credentials — this runs the full adaptation loop against a
deterministic mock black box and prints the Table-2-shaped report:

```bash
python scripts/run_experiment.py --dataset strategyqa --setting ground_truth \
    --limit-train 32 --limit-eval 16 --max-train-steps 50 --dry-run
```

With real Azure credentials and the paper budget:

```bash
python scripts/run_experiment.py --dataset strategyqa --setting ground_truth
```

Inspect the config that a run would use:

```bash
python -c "import bbox_adapter as b; import json; print(json.dumps(b.get_config('gsm8k'), indent=2))"
```

Package self-tests (dependency-light, no network):

```bash
python -m bbox_adapter                # package metadata + utils smoke test
python -m bbox_adapter.llm            # black-box payload guard / temperature routing
python -m bbox_adapter.losses.nce     # Eq. (3) gradient sign structure
python -m bbox_adapter.inference.beam_search
python -m bbox_adapter.eval.metrics
python -m bbox_adapter.eval.cost
python -m bbox_adapter.eval.vram
```

---

## 5. Configuration knobs

Configs are **YAML deep-merges**: `configs/default.yaml` is the base, a per-dataset file
overrides it, and CLI flags override both:

```python
from bbox_adapter import get_config            # default + dataset
cfg = get_config("gsm8k", overrides={"loss": {"alpha": 1e-3}})
```

Paper settings pinned in `default.yaml` / per-dataset YAMLs (Appendix H.2):

| Key | Value | Meaning |
| --- | --- | --- |
| `training.lr` | `5e-6` | AdamW learning rate `eta` |
| `training.batch_size` | `64` | mini-batch gradient steps |
| `training.max_train_steps` | `6000` | total gradient steps |
| `training.weight_decay` | `0.01` | AdamW weight decay |
| `training.betas` | `[0.9, 0.999]` | AdamW betas |
| `beam_search.beam_size` | `3` | beam width `k` |
| `adapter.max_length` / `blackbox.max_len` | `512` | max sequence length |
| `blackbox.temperature` | `1.0` | BBox-Adapter sampling (`0.0` for SFT/CoT, `0.7` for ToxiGen) |
| `loss.alpha` | `0.01` | `alpha * E[g^2]` regularizer (paper-silent → default) |
| `training.n_iterations` | `4` | outer loop `T` |
| `training.n_candidates` | `5` | candidates `M` sampled per query per iteration |
| `training.k_init` | `5` | initial candidates `K` per query |
| `beam_search.n_samples` | `2` | sentence proposals per beam per step `n` |
| `buffer.sel_mode` | `ground_truth` | `ground_truth` \| `ai_feedback` \| `combined` |
| `loss.name` | `nce` | `nce` \| `mlm` (ablation) |

Adapter backbone resolution (`adapter.backbone: null` → resolved from `dataset` + `size`):

| Dataset | 0.1B | 0.3B |
| --- | --- | --- |
| StrategyQA / GSM8K / ScienceQA / ToxiGen | `microsoft/deberta-v3-base` | `microsoft/deberta-v3-large` |
| TruthfulQA | `bert-base-cased` | `bert-base-cased` |

Pooling: CLS for BERT; mask-aware mean for DeBERTa-v3 (paper-silent → documented default).

---

## 6. Method overview & code map

| Paper | Concept | Code |
| --- | --- | --- |
| §3.1 | energy adapter `g_theta(x, y) → R` | `adapter/energy_model.py` |
| §3.2 Eq. (1) | posterior `p_theta(k\|{x_k}) = exp(g(x_k)) / Σ exp(g(x_j))` | `losses/nce.py::softmax_posterior` |
| §3.2 Eq. (2) | ranking NCE: maximize `E_pdata[g(x,y+)] − log Σ_k exp(g(x_k))` | `losses/nce.py::compute_nce_loss` |
| §3.2 Eq. (3) | gradient `−E[g(y+)] + αE[g(y+)²] + E_pθ[g(y−)] + αE[g(y−)²]` | `losses/nce.py::nce_gradient_terms`, `adapter/regularizer.py` |
| §3.2 | "spectral normalization" = **L2 energy penalty** (Addendum; not power iteration) | `adapter/regularizer.py` |
| §3.3 Eq. (4) | sentence-level beam search: LLM proposes, `g_theta` prunes top-`k` | `inference/beam_search.py`, `inference/selector.py` |
| §3.4 Eq. (5)/(6) | positive refresh `y+^(t) = SEL(y+^(t−1), {ŷ_m})`; negatives `{ŷ_m \| ŷ_m ≠ y+^(t)}` | `training/buffers.py` |
| §3.4 Alg. 1 | outer `T` loop, inner dataset pass, `theta ← theta − eta·grad` | `training/online_adaptation.py` |
| §4.1 | GPT-4 simulated human preference `SEL(·)` | `feedback/ai_feedback.py` |
| App. J | generator prompts (2-shot / 4-shot / 1-shot / instruction) | `llm/prompts.py` |
| App. G | rater criteria: Coherency / Reasonability / Correctness / Format | `llm/prompts.py`, `feedback/ai_feedback.py` |
| App. C | black-box contract: **no logprobs / echo / logit_bias ever sent** | `llm/blackbox_client.py::assert_text_only_payload` |

The black-box payload guard is enforced structurally: every request is validated before
being sent and raises `ValueError` if a probability-bearing field appears.

SEL modes (`buffer.sel_mode` / `training.sel_mode`):

* `ground_truth` — pick the candidate matching the dataset gold answer.
* `ai_feedback` — ask `gpt-4` to pick the best candidate (ranked top-5 for TruthfulQA).
* `combined` — ground truth first, GPT-4 preference as augmentation/fallback.

Outcome supervision (on by default): any inference whose final answer matches current
positives joins the positives; the rest join the negatives.

---

## 7. Reproducing the paper

All scripts accept `--dry-run`, `--allow-mock`, `--limit-train/--limit-eval` for fast
offline iteration and write artifacts (JSON/CSV/Markdown) under `runs/`.

### 7.1 Table 2 — main results (adapting gpt-3.5-turbo)

Paper targets (Ground-Truth / AI Feedback / Combined, base model in parentheses):

| Dataset | GT | AI | Combined | Base |
| --- | --- | --- | --- | --- |
| StrategyQA | 71.62 | 69.85 | 72.27 | 66.59 |
| GSM8K | 73.86 | 73.50 | 74.28 | 67.51 |
| TruthfulQA | 79.70 | 82.10 | 83.60 | 77.00 |
| ScienceQA | 78.53 | 78.30 | 79.40 | 72.90 |

Average delta ≈ **+6.39%**, max **+6.77%** (GSM8K, Combined). Combined should be strongest;
AI Feedback should be competitive with Ground-Truth.

```bash
# one cell
python scripts/run_experiment.py --dataset strategyqa --setting combined --size 0.1b

# entire Table 2 (4 datasets × 3 settings × 2 sizes)
python scripts/run_experiment.py --table2 --datasets strategyqa gsm8k truthfulqa scienceqa \
    --settings ground_truth ai_feedback combined --sizes 0.1b 0.3b --seeds 0 1 2
```

Artifacts: `runs/<dataset>_<size>_<setting>/results_raw.json`,
`summary.json`, `REPORT.md`. Multi-seed runs report population standard deviations
(Table 10) and a comparison against the paper anchors with tolerance.

Full-step vs single-step inference is selected with `--inference {full,single_step}`);
single-step asks the LLM for complete answers once and only *ranks* them with `g_theta`.

### 7.2 Table 3 — plug-and-play (no retraining)

The adapter is trained **only** on gpt-3.5-turbo and then transplanted, unmodified, onto
other black boxes. `run_plug_and_play.py` hashes the adapter parameters before and after
evaluation and exits non-zero if anything changed.

Paper targets:

| Black box | StrategyQA | GSM8K | TruthfulQA | Average (Δ) |
| --- | --- | --- | --- | --- |
| davinci-002 | 59.61 (+15.42) | 23.85 | 36.50 | 39.99 (+6.85) |
| Mixtral-8x7B | 63.97 (+4.06) | 47.61 | 49.70 (+9.30) | 53.76 (+4.50) |

```bash
# train the plugger once on gpt-3.5-turbo, then transplant
python scripts/run_experiment.py  --dataset strategyqa --setting combined --size 0.1b
python scripts/run_plug_and_play.py --blackboxes davinci-002 mixtral-8x7b-v0.1 \
    --datasets strategyqa gsm8k truthfulqa --size 0.1b
```

Checkpoints are named `<dataset>_<size>_<blackbox>` (e.g.
`adapter_strategyqa_0.1b_gpt-3.5-turbo_final`) so the plugger identity is explicit;
`--adapter-tag` / `resolve_checkpoint()` let you point at any trained theta.

### 7.3 Table 4 — cost (`$` / 1k questions)

Token usage is metered by a `CostLedger` (`llm/token_cost.py`, `eval/cost.py`) using
**gpt-3.5-turbo-1106** pricing ($0.0015/1k input, $0.0020/1k output).

Paper targets:

| Dataset / mode | Accuracy | Train $ | Inference $/1k |
| --- | --- | --- | --- |
| StrategyQA single-step | 69.87 | 2.77 | 2.20 |
| StrategyQA full-step | 71.62 | 3.48 | 5.37 |
| StrategyQA Azure-SFT | — | 153.00 | 7.50 |
| GSM8K single-step | 71.13 | 7.54 | 3.10 |
| GSM8K full-step | 74.28 | 11.58 | 12.46 |
| GSM8K Azure-SFT | — | 216.50 | 28.30 |

→ ≈ **31.30× cheaper training / 1.84× cheaper inference** (StrategyQA, full-step) vs Azure-SFT.

```bash
python scripts/run_experiment.py --dataset strategyqa --setting combined \
    --inference single_step --cost-report
python scripts/run_experiment.py --dataset strategyqa --setting combined \
    --inference full --cost-report
```

Cost rows land in `runs/.../cost_report.json` and the `REPORT.md` cost table;
`compare_cost_to_paper()` prints a pass/fail check per cell.

### 7.4 Table 5 — NCE vs MLM ablation & scale analysis

Swap only the objective, holding everything else fixed:

| Dataset | Size | MLM | NCE |
| --- | --- | --- | --- |
| StrategyQA | 0.1B | 61.52 | 71.62 |
| StrategyQA | 0.3B | 60.41 | 71.18 |
| GSM8K | 0.1B | 70.56 | 72.06 |
| GSM8K | 0.3B | 70.81 | 73.86 |

```bash
python scripts/run_ablation.py --mode ablation \
    --datasets strategyqa gsm8k --sizes 0.1b 0.3b --losses nce mlm --seeds 0 1 2
```

Artifacts: `runs/ablation/ablation_results.json`, `ablation_runs.csv`, `ABLATION.md`
(with the Table-5 layout and a tolerance check versus the paper).

### 7.5 Figure 3 — beam-size and iteration sweeps

* **Fig. 3(a)** beam sizes `k = 1, 3, 5` → ≈ **+2.41%** average gain across 0.1B/0.3B on StrategyQA.
* **Fig. 3(b)** iterations `T = 0..4` → **degradation at T = 0** (untrained adapter),
  improvement after one round and beyond.

```bash
python scripts/run_ablation.py --mode beams      --datasets strategyqa --beams 1 3 5
python scripts/run_ablation.py --mode iterations --datasets strategyqa --iterations 0 1 2 3 4
```

The T = 0 cell is the paper's key sanity check: an untrained adapter must score **worse**
than the base model. `iteration_analysis()` asserts this relationship and reports it.

### 7.6 Table 6 — Mixtral (white box treated as black box) + VRAM

| Method | StrategyQA acc (0.1B / 0.3B) | Train VRAM | Inference VRAM |
| --- | --- | --- | --- |
| Base Mixtral | 59.91 | — | 90 GiB |
| BBox-Adapter | 66.08 / 65.26 | 105 GiB | 92 GiB |
| SFT-LoRA | — | 208 GiB | 92 GiB |

Per the **Addendum, VRAM is reported only for the 0.1B adapter**; `eval/vram.py`
enforces this via `VramConfig.report_only_0_1b`.

```bash
python scripts/run_plug_and_play.py --mode whitebox --dataset strategyqa --size 0.1b \
    --measure-vram
```

### 7.7 Table 7 — ToxiGen (toxicity reduction)

| Metric | Base Mixtral | BBox-Adapter |
| --- | --- | --- |
| Toxic % | 41.90 | 20.60 |
| Toxicity Prob % | 41.02 | 20.75 |

Lower is better. ToxiGen has no gold continuation, so `SEL(·)` uses GPT-4 AI feedback
(with a deterministic majority fallback offline); generation temperature is 0.7.

```bash
python scripts/run_toxigen.py --blackbox mixtral --size 0.1b --setting ai_feedback
python scripts/run_toxigen.py --self-test          # offline anchor checks
```

Artifacts: `toxigen_results.json`, `toxigen_summary.json`, `toxigen_rows.csv`, `TOXIGEN.md`.

### 7.8 Figure 4 — GSM8K case study

The paper's example: plain CoT answers the "flights to France" question incorrectly, while
the BBox-Adapter sentence-level beam search recovers the correct answer (**11**).
Reproduce by running GSM8K with `--inference full` and inspecting the stored per-query
candidates/energy history:

```bash
python scripts/run_experiment.py --dataset gsm8k --setting combined --inference full \
    --dump-candidates --limit-eval 50
```

`BeamSearchResult.energy_history` and `keep_candidates` retain every beam step so the
winning beam can be printed against the greedy CoT baseline.

### 7.9 Baselines (CoT / Azure-SFT / SFT-LoRA)

* **CoT prompting** of the unadapted black box at temperature `0.0` — always runnable.
* **Azure-SFT**: uses the Azure OpenAI fine-tuning service; only epochs / batch size /
  LR-multiplier are adjustable. We use **3 epochs** per §F.2 and note the §H.2 conflict
  (5 epochs) — see `AZURE_SFT_EPOCH_CONFLICT` in the script.
* **SFT-LoRA** on Mixtral-8x7B-v0.1: `r = 128` (0.1B-equivalent) or `r = 384`
  (0.3B-equivalent), `alpha = 2r`, dropout 0.1, LR 2e-4, wd 0.001, batch 8/GPU,
  max grad norm 0.3, Paged AdamW 32-bit, cosine schedule, 4× A100-80GB.

```bash
# specs + cost anchors only (no external calls)
python scripts/run_baselines_cot_sft.py --mode cot --datasets strategyqa gsm8k --dry-run

# real CoT baseline
python scripts/run_baselines_cot_sft.py --mode cot --datasets strategyqa gsm8k truthfulqa

# gated: submit Azure fine-tuning job / run LoRA training
python scripts/run_baselines_cot_sft.py --mode azure_sft --run-azure-sft
python scripts/run_baselines_cot_sft.py --mode lora_sft  --run-lora-sft --sizes 0.1b 0.3b
```

Artifacts: `baselines_raw.json`, `baselines_summary.json`, `baselines_rows.csv`, `BASELINES.md`.

---

## 8. Running without credentials (offline / mock mode)

Every script defaults to `--allow-mock`, so with no Azure key the `MockLLMClient` produces
deterministically formatted answers (`#### Yes.`, `#### The answer is 7`, refusal text)
and the whole pipeline executes: candidate sampling, SEL, Eq. (5)/(6) buffer refresh,
NCE training, beam search, metrics, cost ledger.

```bash
python scripts/run_experiment.py --dataset strategyqa --setting ground_truth \
    --limit-train 16 --limit-eval 8 --max-train-steps 20
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `--dry-run` | build everything, skip the heavy loops |
| `--allow-mock` / `--no-mock` | toggle graceful fallback to the mock client |
| `--limit-train N` / `--limit-eval N` | subsample splits |
| `--max-train-steps N` | override the 6000-step budget |
| `--seeds 0 1 2` | multi-seed runs with population std reporting |
| `--output-dir runs/foo` | artifact location |

---

## 9. Tests & sanity checks

```bash
pytest tests/ -v
# or standalone (no pytest needed)
python tests/test_nce_loss.py
python tests/test_beam_search.py
python tests/test_buffers.py
python tests/test_answer_extraction.py
```

What the suite pins down:

* **Eq. (1)** posterior normalization, log-softmax consistency, shift invariance, masking.
* **Eq. (2)** loss equals a manual per-query softmax cross-entropy; the loss decreases as
  the positive ranks higher; ragged negative lists and zero-negative sets are handled.
* **Eq. (3)** gradient sign structure — positives get a *negative* gradient, negatives a
  *positive* one, and both regularizer terms are positive; `alpha = 0` is a no-op;
  `alpha·E[g²]` is differentiable with gradient `2·alpha·g`.
* **Regularizer is not power iteration** (source-level guard per the Addendum).
* **Beam search** honours adapter ranking, keeps ≤ beam-size hypotheses, stops on `####`,
  and issues **no** `logprobs` / `echo` / `logit_bias` / `top_logprobs` kwargs.
* **Single-step ranking** performs **zero** LLM calls.
* **Buffers** implement SEL for all three paper settings plus Eq. (5)/(6) refresh and
  outcome supervision (and the disable switch).
* **Answer extraction** round-trips the Appendix-J formats for all five datasets.

Additional validation suggested by the reproduction plan:

1. **T = 0 sanity** — untrained adapter must be worse than the base model (Fig. 3b).
2. **NCE ≫ MLM** with everything else fixed (Table 5, ≈10-point gap on StrategyQA).
3. **Energy curves** — mean positive energy rises and mean negative energy falls over
   training; logged every `training.log_every` steps into `metrics.jsonl` and plotted as
   CSV curves (`write_curve_csv`) for Appendix-K style figures.
4. **Black-box purity** — payload assertions in `blackbox_client.build_payload`.

---

## 10. Defaults for paper-silent hyper-parameters

Where the paper is silent we chose documented defaults (all flagged in the YAMLs and
docstrings), while the paper is authoritative wherever it speaks:

| Item | Paper | Our default | Rationale |
| --- | --- | --- | --- |
| `alpha` (Eq. 3 regularizer) | not specified | `1e-2` (sweep `{1e-3, 1e-2, 1e-1}`) | energy-curve stability (`--mode alphas`) |
| Advisor pooling/head | "small pretrained encoder + scalar head" | CLS (BERT) / mean (DeBERTa-v3) + single `Linear(H,1)` | backbone conventions |
| Candidates `n` per beam per step | not specified | `2` | cost/quality trade-off |
| Initial candidates `K` | not specified | `5` | plan default |
| Candidates `M` per outer iteration | `M` in Algorithm 1 | `5` | plan default |
| Outer iterations `T` | Fig. 3(b) plateaus by ~4 | `4` | matches plateau |
| Beam score length normalization | not specified | none (`normalization: none`) | faithful to "highest-scoring option" |
| Optimizer betas / schedule | AdamW, wd 0.01, lr 5e-6 | `(0.9, 0.999)`, 10% linear warmup + constant | standard, stable |
| Contrastive set | one positive + negatives per query | index 0 = positive, rest = negatives | Eq. (2) |
| Sentence segmentation | `s^1..s^L` sentence-level | split on sentence terminals / newlines | matches "one step per line" prompts |
| Azure-SFT epochs | §F.2 says 3, §H.2 says 5 | `3` | §F.2, conflict documented |

`6000` "training steps" are interpreted as **total mini-batch gradient steps**, distributed
across the `T` outer iterations (remainder given to the last iteration so the total is exact).

---

## 11. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ImportError: transformers` / `torch` | install `requirements.txt`; the tests still run without torch |
| Azure 401 / 404 | check `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `api_version`, and deployment names |
| Out of memory loading Mixtral | use `device_map="auto"` with ≥90 GiB, or point `--blackbox gpt-3.5-turbo` |
| `deberta-v3` tokenizer error | ensure `sentencepiece` is installed |
| Adapter checkpoint not found | `run_plug_and_play.py` requires a trained adapter; pass `--adapter-dir/--adapter-tag` or `--allow-untrained` (for debugging only) |
| Mixtral accuracy looks low | verify the instruction-drop rule: StrategyQA/GSM8K prompts omit the instruction sentence for Mixtral/davinci (`uses_instruction`) |
| TruthfulQA metric seems noisy | True+Info needs the GPT judge; `eval.gpt_judge: false` uses the lexical surrogate |

---

## 12. Citation

```bibtex
@inproceedings{bboxadapter,
  title     = {Lightweight Adapting for Black-Box Large Language Models},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2024}
}
```

If you use this codebase, please cite the original paper and note that this is an
independent reproduction with documented defaults where the paper is silent.
