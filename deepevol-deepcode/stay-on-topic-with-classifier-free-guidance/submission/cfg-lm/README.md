# Stay on Topic with Classifier-Free Guidance — Reproduction

A training-free, **inference-time** implementation of *Classifier-Free Guidance* (CFG)
(Sanchez et al., 2023 — [arXiv:2306.17806](https://arxiv.org/abs/2306.17806)) for
autoregressive language models.

At every decoding step two forward passes are run **through the same LM weights**:

| pass | context |
|------|---------|
| conditional   | the prompt `c` (or `c` + generated tokens) |
| unconditional | the prefix dropped (`empty_prefix`), or the context starting at the last prompt token (`last_prompt_token`, the §3.1 harness convention), or a **negative prompt** `c̄` (Eq. 5) |

The next-token logits are then combined in **log space** before any temperature / top-p
/ softmax step:

```
guided = uncond + gamma * (cond - uncond)          # Eq. 7 (paper §2.2)
```

* `gamma = 1.0` → exact vanilla conditional decoding (identity, asserted in tests)
* `gamma = 0.0` → exact unconditional decoding
* `gamma > 1` → sharpens the prompt-conditional direction, lowers sampling entropy,
  improves prompt adherence, and emulates a model of roughly twice the size *at equal
  inference FLOPs* (two forward passes per token).

---

## 1. Repository layout

```
cfg-lm/
├── src/
│   ├── cfg/                    # core CFG (Phase 1 — underlies everything)
│   │   ├── logits.py           # Eq. 7 / Eq. 5 combination + softmax/top-p primitives
│   │   ├── model_wrapper.py    # HF dual-context forward pass (same weights, 2× kv-cache)
│   │   ├── sampler.py          # CFG -> temperature -> top-p -> softmax -> multinomial
│   │   └── generator.py        # autoregressive CFG decode loop
│   ├── eval/
│   │   ├── harness_cfg.py      # EleutherAI LM-Eval-Harness CFG shim (Table 5)
│   │   ├── triviaqa_match.py   # TriviaQA substring-match scoring (Appendix C.1)
│   │   ├── cot_eval.py         # CoT valid-answer parsing + accuracy (Figs 2 / 17)
│   │   ├── pass_at_k.py        # unbiased pass@k (Chen et al. 2021)
│   │   └── humaneval_eval.py   # sandboxed code execution harness
│   ├── analysis/
│   │   ├── entropy.py          # §5.1 H(p) per token, mean over completion (4.7 vs 5.49)
│   │   ├── overlap.py          # §5.2 top-p=0.9 token overlap (~50%) / Spearman (r_s > .7)
│   │   ├── perplexity.py       # §5.2 continuation-only PPL correlations (.94 / .70)
│   │   ├── visualize.py        # §5.3 Table 3 vocabulary re-ranking
│   │   ├── flops.py            # §4.1 ELECTRA-style FLOPs per token (CFG = 2×)
│   │   └── ancova.py           # §4 logistic regression + ANCOVA (p = .01, Table 6)
│   └── data/
│       ├── p3_sampler.py       # 32,902-datapoint P3 subsample (~50/dataset, >200-token drop)
│       └── prompts.py          # CoT few-shot prompts, neg/pos system prompts, gamma grids
├── scripts/
│   ├── run_zero_shot.py        # Table 5 sweep
│   ├── run_cot.py              # Figures 2 / 17
│   ├── run_humaneval.py        # Tables 2 / 7 / 8 / 9
│   ├── run_analysis.py         # Section 5 (entropy / overlap / PPL / Table 3)
│   └── run_flops.py            # Section 4.1 + Table 6
├── configs/default.yaml        # single source of truth: gammas, temps, modes, anchors
├── tests/test_cfg.py           # unit tests for every equation + pipeline
├── requirements.txt
└── README.md
```

Every paper constant (gamma grid, temperatures, prompt modes, paper anchor values) lives in
`configs/default.yaml` and is mirrored by module-level constants, so **Table 5, Figures 2/17
and Tables 2/7/8/9 all share one code path**.

---

## 2. Installation

Python 3.10+ is required.

```bash
cd cfg-lm
python -m venv .venv && source .venv/bin/activate

# CUDA 12.1 build of torch (swap the index URL for your CUDA version, or use CPU wheels)
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

`requirements.txt` pins `lm-eval==0.4.2` because the Table 5 numbers depend on the
**unmodified harness defaults** (greedy, `temperature=0.0`, `top_p=1.0`) — CFG is applied to
the raw logits *before* those defaults are applied.

### Minimal / CPU-only install

`torch`, `transformers`, `datasets`, `scipy`, `statsmodels`, `matplotlib` are all optional for
the pure-math layers, which run (and are unit-tested) with only `numpy`:

* `src/cfg/logits.py`, `src/eval/pass_at_k.py`, `src/eval/triviaqa_match.py`,
  `src/eval/cot_eval.py`, `src/data/prompts.py`, `src/analysis/flops.py`,
  `src/analysis/ancova.py`

Every script therefore has an offline validation mode — see `--math-only` / `--dry-run`
below — so the whole pipeline can be smoke-tested without a GPU or network.

### Run the tests

```bash
python -m pytest tests/test_cfg.py -q
```

The suite asserts the paper's identities, including: `gamma=1` ≡ conditional, `gamma=0` ≡
unconditional, consistent Eq. 7 across `cfg.logits` / `cfg.sampler` / `eval.harness_cfg`,
sampling-stage ordering (`CFG → temperature → top-p → softmax → multinomial`), the unbiased
pass@k estimator, the §5 metric anchors (entropy 4.7 / 5.49, overlap ≈ 0.5, PPL corr
0.94 / 0.70), the §4.1 2× FLOPs multiplier, and the Table 6 split
(5 inconclusive / 2 CFG / 2 vanilla).

---

## 3. Models and datasets

Model weights are pulled from the HuggingFace hub on first use. Nothing needs manual
downloading except that you need disk space (and a GPU for the 7B+ models).

| purpose | models |
|---------|--------|
| Table 5 zero-shot | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`; `EleutherAI/pythia-160m` … `pythia-12b` |
| HumanEval (Tables 2/7/8/9) | `Salesforce/codegen-350M-mono`, `codegen-2B-mono`, `codegen-6B-mono` |
| Section 5 analysis | `tiiuae/falcon-7b` (base) + `tiiuae/falcon-7b-instruct` |
| CoT (Figs 2/17) | `WizardLM-30B`, `Guanaco-65B` (needs A100-class memory or sharding) |

Datasets are streamed from the hub: `ai2_arc`, `google/boolq`, `hellaswag`, `piqa`,
`allenai/sciq`, `mandarjoshi/trivia_qa`, `allenai/winogrande`, `EleutherAI/lambada_openai`,
`openai/gsm8k`, `nguyen-brat/aqua`, `bigscience/P3`, `openai_humaneval`.

**Hardware.** One 24 GB GPU is enough for 7B models in fp16. CFG doubles the kv-cache per
token (2C) and doubles inference FLOPs per token — budget accordingly.

`LLaMA` models are **out of scope** for this reproduction and are never downloaded.

---

## 4. How to run each paper result

All scripts accept `--help`. Common flags: `--models`, `--tasks`, `--gammas`,
`--output`/`--out`, `--limit`, `--seed`, `--dry-run`, `--math-only`.

### 4.1 Table 5 — zero-shot gamma sweep (Phase 2)

```bash
# Full Table 5: GPT-2 family + Pythia family, 9 tasks, gamma grid
python scripts/run_zero_shot.py \
    --models gpt2,gpt2-medium,gpt2-large,gpt2-xl \
    --tasks arc_challenge,arc_easy,boolq,hellaswag,piqa,sciq,triviaqa,winogrande,lambada_openai \
    --gammas 1.0,1.1,1.25,1.5,1.75,2.0 \
    --output outputs/zero_shot_gpt2.json

# One model, first sanity check on the direction of the effect
python scripts/run_zero_shot.py --models gpt2-large --tasks lambada_openai --limit 200

# Offline pipeline check (deterministic synthetic numbers, no GPU / no harness)
python scripts/run_zero_shot.py --dry-run
```

The harness convention for this suite is **`unconditional_mode = last_prompt_token`** (the
unconditional prompt begins at the last token of the initial prompt, §3.1). TriviaQA is scored
by **substring matching** (`src/eval/triviaqa_match.py`, Appendix C.1).

Expected anchors printed by `--anchor-report`:

| model | task | vanilla (γ=1) | CFG |
|-------|------|---------------|-----|
| GPT-2-large | LAMBADA | 47.7 | 60.5 |
| Pythia-12B | LAMBADA | 70.4 | 80.6 |
| Pythia-6.9B | SciQ | ~84.3 | ~89.7 |

CFG (especially γ≈1.5) improves most tasks, while ARC-c and WinoGrande degrade.
The JSON report is consumed directly by `scripts/run_flops.py`.

### 4.2 HumanEval — Tables 2 / 7 / 8 / 9 (Phase 3)

```bash
# Full sweep: CodeGen-350M/2B/6B-mono x temperature {0.2,0.6,0.8} x gamma grid, k in {1,10,100}
python scripts/run_humaneval.py \
    --models codegen-350M-mono,codegen-2B-mono,codegen-6B-mono \
    --temperatures 0.2,0.6,0.8 \
    --gammas 1.0,1.1,1.25,1.5,1.75,2.0 \
    --n-samples 200 --output outputs/humaneval_report.json

# Deterministic offline smoke test of the sweep + pass@k math
python scripts/run_humaneval.py --dry-run
python scripts/run_humaneval.py --math-only
```

`n = 200` samples per problem (enough for k = 100), 512-token budget, stop strings
`("\nclass", "\ndef", "\n#", "\nif", "\nprint")`, 3 s execution timeout per problem. The
unbiased estimator is `pass@k = 1 - C(n-c, k) / C(n, k)` (Chen et al. 2021).

Expected: pass@1 rises for γ ∈ [1, 1.5] and falls past it, while high-k pass rates flatline or
drop — e.g. CodeGen-350M at temperature 0.2: pass@1 11.0 → ~11.8 at γ=1.1, pass@100 falls from
22.0. `direction_check` and `task_level_vs_vanilla` (Figure 3 semantics: outperform / tie /
underperform counts) are printed at the end.

### 4.3 Chain-of-Thought — Figures 2 / 17 (Phase 3)

```bash
python scripts/run_cot.py \
    --models WizardLM-30B,Guanaco-65B \
    --tasks gsm8k,aqua \
    --gammas 1.0,1.1,1.25,1.5,1.75,2.0 \
    --limit 200 --output outputs/cot_report.json

python scripts/run_cot.py --dry-run        # mock generator, CPU
python scripts/run_cot.py --math-only      # validate chain parsing / scoring only
```

GSM8K uses the **8-shot Self-Consistency** prompt and the `####` answer marker; AQuA uses the
standard few-shot prompt and the `The answer is` marker (both from `src/data/prompts.py`).
Each script run emits two curves per task:

* top panel: **accuracy vs gamma**
* bottom panel: **% of chains ending in a valid, parsable answer vs gamma**

Expected: small γ increases the share of valid chains and raises accuracy; γ > 1.5 degrades
the chains (invalid-chain rate climbs). Artifacts: `cot_report.json`,
`cot_accuracy_<task>.png`, `cot_invalid_<task>.png`.

### 4.4 Section 5 — entropy / overlap / perplexity / Table 3 (Phase 4)

```bash
# Full run on Falcon-7b-Base (+ -Instruct) over the 32,902-datapoint P3 sample
python scripts/run_analysis.py \
    --model tiiuae/falcon-7b \
    --instruct tiiuae/falcon-7b-instruct \
    --gamma 1.5 --max-new-tokens 128 \
    --output outputs/analysis_report.json

python scripts/run_analysis.py --dry-run     # deterministic mock LM (CPU)
python scripts/run_analysis.py --math-only   # pure-NumPy validation of the metric layer
```

What is reproduced:

| metric | expectation |
|--------|-------------|
| mean completion-token entropy `(1/n) Σ H(p(x_i | x_<i))` | ~5.49 vanilla → **~4.7 CFG (γ=1.5)** |
| top-p = 0.9 token-set overlap, CFG vs vanilla | **≈ 50 %** |
| Spearman r_s CFG vs instruction-tuned (longer/harder prompts) | **> 0.7** |
| PPL correlation `PPL_cfg` vs `PPL(y|x)` | **≈ 0.94** |
| PPL correlation `PPL_instruct` vs `PPL_cfg` | **≈ 0.70** |

Perplexity is always computed on the **continuation only** (prompt loglikelihood ignored).

P3 sampling protocol (`src/data/p3_sampler.py`): take ~50 examples from each of the 660 P3
datasets (whole dataset when smaller), drop inputs longer than 200 tokens, deterministic seed
1234, target **32,902 datapoints**. Falcon is the only model pair used on P3. Results are
cached at `data/cache/p3_sample.json`; `sample_p3_synthetic()` provides an offline stand-in.

The §5.3 Table 3 vocabulary re-ranking (prompt
`"The dragon flew over Paris, France"`, `c̄ = ∅`) is produced by the same script:

```python
from src.analysis.visualize import run_table3_walkthrough, format_table3, save_rankings
rankings = run_table3_walkthrough(wrapper, gamma=1.5, n_steps=12, top_k=5)
print(format_table3(rankings))
save_rankings("outputs/table3_rankings.json", rankings)
```

At each step the vocabulary is ranked by `log P_cfg(w) − log P_ref(w)`, and the top-5 /
bottom-5 columns are emitted: prompt-relevant tokens (`dragon`, `dragons`, `flew`, `Paris`,
`France`) are upweighted while unrelated countries / dates / topics (`Queensland`, `1913`,
`hostages`, `voyages`) are downweighted.

### 4.5 Section 4.1 + Table 6 — FLOPs and ANCOVA (Phase 5)

```bash
# Uses the Table 5 report produced above
python scripts/run_flops.py --results outputs/zero_shot_gpt2.json --output outputs/flops_report.json

python scripts/run_flops.py --synthetic       # deterministic accuracy grid, offline
python scripts/run_flops.py --paper-only      # emit the Table 6 anchors directly
```

`src/analysis/flops.py` reimplements the ELECTRA per-token FLOPs estimate (GPT-2, Pythia,
CodeGen, Falcon, WizardLM, Guanaco specs are registered). CFG (γ ≠ 1) gives exactly **2×**
inference FLOPs per token, which is what places a CFG-1.4B model on the accuracy-vs-FLOP line
of a ~2.8B vanilla model.

`src/analysis/ancova.py` then fits a logistic regression of accuracy on **log FLOPs per token**
separately for the CFG group (γ > 1) and the vanilla group (γ = 1), and runs an ANCOVA at
**p = .01** (Rutherford 2011). Reproduced Table 6 direction (5/9 insignificant, 2 favor CFG,
2 favor vanilla):

| task | p-value | winner |
|------|---------|--------|
| LAMBADA (OpenAI) | 0.000 | CFG |
| WinoGrande | 0.003 | Vanilla |
| SciQ | 0.008 | CFG |
| TriviaQA | 0.008 | Vanilla |
| HellaSwag | 0.012 | — |
| PiQA | 0.030 | — |
| ARC-c | 0.216 | — |
| BoolQ | 0.345 | — |
| ARC-e | 0.355 | — |

Artifacts: `flops_report.json`, `ancova.json`, `flops.json`, `accuracy_vs_flops.png`,
`ancova_pvalues.png`.

---

## 5. Switching the unconditional prompt

The unconditional context is controlled by `unconditional_mode` and is also exposed in
`configs/default.yaml` (`unconditional.default` / `unconditional.zero_shot`):

| mode | meaning | used by |
|------|---------|---------|
| `last_prompt_token` | unconditional prompt starts at the **last token of the initial prompt** (§3.1) | zero-shot harness sweep (Table 5) |
| `empty_prefix` | the prompt prefix is dropped entirely (BOS-seeded if empty) | CoT, HumanEval, all Section 5 analyses |

From the CLI:

```bash
python scripts/run_zero_shot.py --unconditional-mode last_prompt_token
python scripts/run_analysis.py --unconditional-mode empty_prefix
```

In code:

```python
from src.cfg import CFGModelWrapper, CFGGenerator, GenerationConfig

wrapper = CFGModelWrapper("gpt2-large", unconditional_mode="last_prompt_token", dtype="auto")
gen = CFGGenerator(wrapper)
out = gen.generate(
    prompts=["Q: What is the capital of France?\nA:"],
    config=GenerationConfig(gamma=1.5, temperature=0.0, max_new_tokens=64),
)
print(out.completions[0])
```

**Negative prompting** (`negative_prompt_logits`, Eq. 5) replaces the unconditional pass with a
negative prompt `c̄` instead of a dropped prefix:

```python
from src.data.prompts import negative_prompt_pair
cond, negative = negative_prompt_pair("a sad system prompt")
out = gen.generate(prompts=[cond_prompt], negative_prompts=[negative],
                   config=GenerationConfig(gamma=3.0))
```

The §3.4 system-prompt study (25 system prompts × 46 user prompts = 1,740 pairs, gammas up to
6.0) is **not run** here; `src/data/prompts.py` ships the prompt constants for completeness.

---

## 6. Canonical hyper-parameters

| setting | value | source |
|---------|-------|--------|
| gamma grid | 1.0, 1.1, 1.25, 1.5, 1.75, 2.0 | §3.1 / §3.3 |
| CFG application point | raw pre-softmax logits, **before** temperature / top-p / softmax | §2.2 |
| zero-shot sampling | greedy (`temperature=0.0`, `top_p=1.0`) — harness defaults | §3.1 |
| HumanEval temperatures | 0.2, 0.6, 0.8 | §3.3.1 |
| HumanEval budget | 512 new tokens, n = 200 samples/problem, timeout 3 s | §3.3.1 |
| CoT budget | 512 new tokens, stop at `####` / `The answer is` | §3.2 |
| analysis gamma | 1.5 (top-p = 0.9, 128 new tokens) | §5 |
| P3 sample | ~50/dataset, 660 datasets, ≤200 input tokens, 32,902 total, seed 1234 | §5 |
| significance | p = .01 | §4 |
| CFG FLOPs multiplier | 2.0 per token (2C kv-cache) | §2.2 / §4 |

Sampling order is fixed by `src/cfg/sampler.py`:
**raw CFG combination → ÷ temperature → top-p nucleus → softmax → multinomial (or argmax for
greedy)**.

---

## 7. Out of scope

Not implemented by this reproduction: LLaMA models, the §3.4 human evaluation study, the §4
VRAM / memory analysis (Eqs. 8–10, Figure 10), Appendix D.1 GPT-J/CodeGen image-generation
experiments, Table 1, and Figures 6, 7, 9, 11, 12, 13.

---

## 8. Reference

```bibtex
@article{sanchez2023stay,
  title   = {Stay on Topic with Classifier-Free Guidance},
  author  = {Sanchez, Guillaume and Fan, Honglu and Spangler, Will and Kalmbach, Patrick and
             Kraus, Peter and Song, Yunmo and Mialon, Gregoire and Dukes, Jordan and
             Kambadur, Myle and Sridhar, Anirudh},
  journal = {arXiv preprint arXiv:2306.17806},
  year    = {2023}
}
```
