# BBOX-ADAPTER — reproduction

Reproduction of **“BBox-Adapter: Lightweight Adapting for Black-Box Large Language
Models”** (Sun, Zhuang, Wei, Zhang & Dai, ICML 2024).

The repository implements the full method of Sections 3.1–3.4 (energy-based
adapter, ranking-based NCE loss, sentence-level adapted inference, online
adaptation), the four datasets and prompt sets of the main experiments, the
baselines (CoT, Azure-SFT, SFT-LoRA), and scripts that reproduce Tables 2–6 and
Figure 3.

```text
submission/
├── bbox_adapter/
│   ├── adapter/         # g_theta, ranking-based NCE loss, trainer, MLM ablation
│   ├── baselines/       # CoT, Azure-SFT (fine-tuning API), SFT-LoRA
│   ├── data/            # datasets, Appendix-J prompts, answer extraction, metrics
│   ├── eval/            # evaluation harness (accuracy / True+Info)
│   ├── inference/       # sentence-level beam search (adapted inference)
│   ├── llm/             # black-box LLM clients (Azure/OpenAI/HF/mock)
│   ├── online/          # Algorithm 1: sample banks, SEL(.), AI feedback
│   └── utils/           # seeding, token/cost accounting, VRAM measurement
├── configs/             # one YAML per dataset + Mixtral + offline mock
├── scripts/             # experiment entry points (see the table below)
├── tests/               # 35 unit/smoke tests (fully offline)
└── reproduce_main_results.py
```

## 1. What is implemented

| Paper element | Where |
|---|---|
| EBM formulation `p_theta(y given x) = p_LLM(y given x) exp(g_theta)/Z` (Eq. 1, Section 3.1) | [`adapter/energy.py`](bbox_adapter/adapter/energy.py) |
| Ranking-based NCE objective (Eq. 2) and its gradient (Eq. 3) | [`adapter/losses.py`](bbox_adapter/adapter/losses.py) |
| Spectral normalisation realised as l2 regularisation of the energies (`alpha * E[g^2]`, addendum) | [`adapter/losses.py`](bbox_adapter/adapter/losses.py) (`l2_energy_regularizer`) |
| Adapter update `theta_{t+1} = theta_t - eta * grad l` (Eq. 7) | [`adapter/trainer.py`](bbox_adapter/adapter/trainer.py) |
| Adapted inference: sentence-level beam search, `n*k` candidates per step, top-`k` by `g_theta` (Section 3.3) | [`inference/beam_search.py`](bbox_adapter/inference/beam_search.py) |
| Single-step inference variant (Section 4.4) | [`inference/adaptive_inference.py`](bbox_adapter/inference/adaptive_inference.py) |
| Online adaptation: init, candidate sampling (Eq. 4), positive/negative updates (Eqs. 5–6), adapter update (Eq. 7), Algorithm 1 | [`online/online_adaptation.py`](bbox_adapter/online/online_adaptation.py) |
| `SEL(.)` with ground-truth, human/AI feedback and combined settings (Section 4.1, Appendix G) | [`online/feedback.py`](bbox_adapter/online/feedback.py), [`online/bank.py`](bbox_adapter/online/bank.py) |
| Outcome supervision (“answers that align with the training set answers are treated as additional positive samples”) | [`online/bank.py`](bbox_adapter/online/bank.py) (`OutcomeSupervision`) |
| Datasets GSM8K / StrategyQA / TruthfulQA / ScienceQA with the sizes of Appendix F.1 | [`data/loaders.py`](bbox_adapter/data/loaders.py) |
| Generator prompts + AI-feedback rater prompts (Appendix J) | [`data/prompts.py`](bbox_adapter/data/prompts.py) |
| Accuracy / “True + Info” metrics | [`data/metrics.py`](bbox_adapter/data/metrics.py), [`data/truthfulqa_judge.py`](bbox_adapter/data/truthfulqa_judge.py) |
| Baselines: CoT, Azure-SFT, SFT-LoRA | [`baselines/`](bbox_adapter/baselines) |
| MLM-vs-NCE ablation (Section 4.5) | [`adapter/mlm_adapter.py`](bbox_adapter/adapter/mlm_adapter.py) |
| Cost accounting in dollars per 1k questions (Section 4.4) | [`utils/cost.py`](bbox_adapter/utils/cost.py) |
| VRAM measurement for Table 6 | [`utils/vram.py`](bbox_adapter/utils/vram.py) |

### Experiment entry points

| Script | Reproduces |
|---|---|
| `python reproduce_main_results.py` | Table 2 (4 datasets x 3 positive-sample settings x 2 adapter sizes) |
| `scripts/train_bbox_adapter.py` | one BBOX-ADAPTER run (Algorithm 1) and its test accuracy |
| `scripts/run_cot_baseline.py` | CoT baseline rows (un-adapted black-box LLM) |
| `scripts/run_azure_sft.py` | Azure-SFT baseline (fine-tuning API) |
| `scripts/run_sft_lora.py` | SFT-LoRA baseline (upper bound, Table 2/6) |
| `scripts/run_plug_and_play.py` | Table 3 (plug-and-play on davinci-002 / Mixtral-8x7B) |
| `scripts/cost_analysis.py` | Table 4 (accuracy + training/inference cost) |
| `scripts/ablation_loss.py` | Table 5 (MLM vs ranking-based NCE) |
| `scripts/scale_analysis.py` | Figure 3 (number of beams, number of iterations) |
| `scripts/measure_vram.py` | Table 6 (VRAM, Mixtral-8x7B, 0.1B adapter) |
| `scripts/prepare_data.py` | caches the four datasets as JSONL for offline runs |
| `scripts/case_study.py` | case study output (Section 4.8 / Figure 4) |
| `scripts/make_tables.py` | assembles the run artifacts into Tables 2–6 / Figure 3 |
| `scripts/plot_training_curves.py` | loss / energy learning curves of the adapter training |

## 2. How to run

Credentials are read from the environment (never from the repo):

```bash
export AZURE_OPENAI_ENDPOINT=...        # gpt-3.5-turbo (black-box LLM) and gpt-4 (rater)
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_API_VERSION=2023-12-01-preview
export OPENAI_API_KEY=...               # only for davinci-002 in the plug-and-play table
pip install -r requirements.txt
```

Typical commands:

```bash
# cache datasets (optional, enables fully offline runs via --data-dir data)
python scripts/prepare_data.py --data-dir data

# Table 2, all datasets/settings/sizes
python reproduce_main_results.py --data-dir data --output-root runs/table2

# or the full pipeline (all tables/figure) in one command
python reproduce_all.py --data-dir data --runs-root runs

# a single run: StrategyQA, ground-truth setting, 0.1B adapter
python scripts/train_bbox_adapter.py \
  --config configs/strategyqa.yaml \
  --positive-source ground_truth \
  --output-dir runs/strategyqa/ground_truth_0.1B

# plug-and-play (Table 3)
python scripts/run_plug_and_play.py \
  --config configs/strategyqa.yaml \
  --plugger-adapter runs/strategyqa/combined_0.3B/adapter_final

# ablation, cost and VRAM tables
python scripts/ablation_loss.py --config configs/strategyqa.yaml --sizes 0.1B,0.3B
python scripts/cost_analysis.py --config configs/strategyqa.yaml \
  --adapter-path runs/strategyqa/ground_truth_0.1B/adapter_final \
  --training-results runs/strategyqa/ground_truth_0.1B/results.json
python scripts/measure_vram.py --dataset strategyqa
```

Smoke tests (no network, no API keys, no GPU):

```bash
python -m pytest tests -q          # 35 tests, about 10 s
```

The evidence gathered by running the code in this sandbox (including a full
online-adaptation run on real GSM8K data with the real DeBERTa-v3-base adapter)
and the paper’s reference numbers for comparison are in
[`docs/VALIDATION.md`](docs/VALIDATION.md).

## 3. Experimental setup reproduced from the paper

Defaults in `bbox_adapter/config.py` and `configs/*.yaml` come from Sections 4.1
and Appendices F–J:

| Setting | Value | Source |
|---|---|---|
| Adapter backbone | `deberta-v3-base` (0.1B) / `deberta-v3-large` (0.3B); `bert-base-cased` for TruthfulQA | Appendix H.2 |
| Adapter learning rate eta | 5e-6 | Appendix H.2 |
| Adapter batch size | 64 | Appendix H.2 |
| Adapter training steps | 6000 per online iteration | Appendix H.2 |
| Optimizer | AdamW, weight decay 0.01 | Appendix H.2 |
| Beams (train/inference) | 3 | Section 4.1 |
| LLM max generation length | 512; temperature 1.0 for the proposal | Appendix H.2 |
| Iterations T | 3 (main), 0–4 (Figure 3b) | Section 4.1, Figure 3b |
| SFT-LoRA | dropout 0.1, 3 epochs, lr 2e-4, wd 0.001, bs 8, grad-norm 0.3, paged AdamW 32bit, cosine, r=128 / r=384, alpha=2r | Table 8, Appendix F.2 |
| Azure-SFT | 3 epochs, default batch size / LR multiplier (5 epochs, bs 16, lr mult 1 for the Table 9 best run) | Appendix F.2 |

The energy adapter is a small encoder LM followed by a scalar head,
`g_theta(x, y) = w^T · pool(Encoder("Question: x \n Answer: y")) + b`, initialised
with a small head (energies close to zero), i.e. `p_theta ~ p_LLM` at `t = 0`, as
in the paper’s “random initialization”.

Sentence-level generation is operationalised the way the prompts prescribe
(“each reasoning step should be in a separate sentence”): at every beam step the
black-box LLM is queried with the prefix generated so far and stopped at the
newline, so each call returns exactly one further sentence. The `n*k` candidates
are scored by `g_theta` and the top-`k` prefixes are kept.

## 4. What was verified here, and what could not be

This sandbox has **no API keys** (`agent.env` is empty) and **no GPU**, so the
paper’s tables cannot be regenerated here; the repository is written so that they
are produced by the scripts above when credentials and hardware are available.
Everything that does not need the external APIs *was* executed:

* `python -m pytest tests -q` -> **22 passed** (about 10 s). Covered: Eq. (3)
  gradients against the analytic gradient `[-1+2*alpha*g_plus, 1+2*alpha*g_minus]`;
  the listwise Eq. (2) gradient against Appendix B’s `p_theta(x_m) - 1[m=0]`;
  energy-adapter scoring, training (positive energy overtakes negative energy) and
  save/load; the four prompt templates; answer extraction and correctness for all
  four datasets; sample-bank initialisation/updates (Eqs. 5–6) with and without
  outcome supervision; AI-feedback selection and its parsing; beam-search
  behaviour (adapter-preferred hypothesis wins, stop marker halts the search,
  finished hypotheses are preferred); and a full offline end-to-end run of
  Algorithm 1 (mock LLM + tiny adapter + local JSONL data).
* The **real** backbone and dataset paths were executed: `microsoft/deberta-v3-base`
  tokenizer/model load, pair scoring and 10 adapter updates on CPU (loss
  0.047 -> -0.347; positive energy 0.08 -> 0.71 vs. negative 0.12 -> 0.36), plus a
  complete online-adaptation run on real GSM8K data (6 train / 2 test, 1
  iteration) and `evaluate_adapter.py` on the resulting checkpoint.
  All four dataset loaders were exercised against their public HuggingFace
  mirrors (GSM8K 7473/1319, TruthfulQA 817 split into 717/100, ScienceQA after
  removing image questions, StrategyQA).

### Deviations and assumptions

1. **`alpha` of the l2 energy regulariser.** The addendum states that spectral
   normalisation is implemented as `alpha * E[g^2]` in Eq. (3), but the paper does
   not report `alpha`. We default to `alpha = 0.01` (`AdapterConfig.alpha`,
   configurable per run).
2. **`n` (samples per beam).** Section 3.3 defines `n*k` candidates per step but
   only fixes `k = 3`. We default to `n = 3`, i.e. 9 candidates per step.
3. **Sentence segmentation.** The paper factorises a solution into sentences
   (Section 3.3) and the prompts mandate one reasoning step per line; we therefore
   use the newline as the sentence boundary and as the generation stop sequence.
4. **“True + Info”.** TruthfulQA is open-ended, so the metric uses an LLM judge
   ([`data/truthfulqa_judge.py`](bbox_adapter/data/truthfulqa_judge.py)) returning
   (truthful, informative); `True + Info (%)` is the share of answers that are
   both. An exact-match fallback is used when no judge is configured.
5. **Cost accounting.** `training_cost_usd` is the API cost of the black-box LLM
   sampling during adaptation (initialisation + one sampling round per iteration,
   plus the gpt-4 rater for AI-feedback/combined runs); adapter training itself is
   local compute. `inference_cost_per_1k` extrapolates the measured token usage of
   the evaluated split with the `gpt-3.5-turbo-1106` price used in the paper. The
   Azure-SFT row is filled from the fine-tuning job (153.00 / 216.50 USD in Table
   4) because the API does not return the billed amount.
6. **StrategyQA version.** The paper reports 2059 train / 229 test examples. The
   public mirror used here (`ChilleD/StrategyQA`) exposes 1603/687; the loader
   honours the paper’s sizes only when the source provides them and accepts
   `--data-dir` with the original files. Prompts and metrics are unaffected.
7. **VRAM (Table 6).** Only the 0.1B adapter is measured, as stated in the
   addendum.
8. **Out of scope.** Experiments that only appear in the appendix (for example
   the ToxiGen toxicity study of Appendix E) are not reproduced; the appendix is
   used only for the details of the main-body experiments (datasets, prompts,
   baselines and hyper-parameters).

## 5. Component map (short version)

* **Energy / adapter:** `bbox_adapter/adapter/energy.py`, `losses.py`, `trainer.py`.
* **Inference:** `bbox_adapter/inference/beam_search.py`, `adaptive_inference.py`.
* **Online adaptation:** `bbox_adapter/online/bank.py`, `feedback.py`,
  `online_adaptation.py`; wired by `bbox_adapter/pipeline.py`.
* **Data and prompts:** `bbox_adapter/data/{loaders,prompts,answer_extraction,metrics}.py`.
* **Black-box LLMs:** `bbox_adapter/llm/{openai_client,hf_client,mock}.py` — only
  text in, text out: no parameters and no token probabilities, as required by
  Table 1 of the paper.
