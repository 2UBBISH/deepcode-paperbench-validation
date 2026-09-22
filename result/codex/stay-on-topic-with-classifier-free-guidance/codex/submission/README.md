# Stay on topic with Classifier-Free Guidance — reproduction

This repository reproduces the core contributions of

> **Stay on topic with Classifier-Free Guidance**
> G. V. Sanchez, A. Spangher, H. Fan, E. Levi, S. Biderman. ICML 2024.

Classifier-Free Guidance (CFG), which previously only worked in text-to-image
models, is applied here to the **logits of an autoregressive language model**
at inference time, with no extra training.

---

## 1. The method this repository implements

Everything is built on the paper's Equation 7. For a prompt/conditioning `c`
and a continuation `w = w_1 … w_T`, the gamma-reweighted next-token
distribution

```
log P(w_i | w_<i, c) = log P(w_i | w_<i)
        + gamma * ( log P(w_i | w_<i, c) - log P(w_i | w_<i) )      (Eq. 7)
```

is equivalent, up to a per-step additive constant that does not change the
resulting softmax, to the logit-space mixture

```
logits_cfg = (1 - gamma) * logits_uncond + gamma * logits_cond
```

which is exactly the "take a step of size gamma away from the unconditional
vector in the direction of the conditioning" operation (Eq. 4).  `gamma = 1`
recovers vanilla prompting, `gamma = 0` the unconditional model, and
`gamma > 1` over-emphasises the prompt.

Two details from the paper that are implemented explicitly:

* **Dropping the prompt is natural for a language model** (Section 2.2): the
  unconditional branch is produced by the same model, with the prompt
  dropped.  Following Section 3.1 we keep the *last token of the prompt* as a
  minimal context, and the number of retained tokens is configurable
  (`uncond_prefix_tokens`, default `1`).
* **Negative prompting** (Eq. 5) generalises the "unconditional" point to an
  arbitrary negative conditioning `c_bar`; the same code path is reused with
  `negative_prompt=...`.  `c_bar = empty` recovers Eq. 7.

The implementation lives in `cfglm/cfg.py` (the maths), `cfglm/generation.py`
(a HuggingFace `LogitsProcessor` for sampling, plus a generation helper) and
`cfglm/scoring.py` (likelihood-based scoring, which is what the benchmark
tables use).

---

## 2. Repository layout

```
cfglm/                        the CFG library
  cfg.py                      Equations 4/5/7 in logit and log-prob space
  generation.py               CFGLogitsProcessor + cfg_generate (+ negative prompting)
  scoring.py                  CFGScorer: per-token CFG log-probabilities for benchmarks
  harness.py                  zero-shot evaluation loop (acc / acc_norm / substring match)
  tasks.py                    the nine zero-shot benchmarks of Table 5
  models.py                   the model families used in the paper
  flops.py                    inference FLOP accounting (ELECTRA convention, Section 4)
  stats.py                    entropy, top-p overlap, Spearman, pass@k, ANCOVA
  humaneval.py                HumanEval sampling + sandboxed unit-test execution
  cot.py                      Chain-of-Thought prompts, answer parsing, self-consistency
  p3.py                       sampling the Section 5 dataset from P3
  distributions.py            P(y|x) / P(x) / CFG / instruct distributions + Table 3 ranking
  external_classifiers.py     the sentiment/toxicity classifiers of Table 4
  fudge.py                    the FUDGE external-classifier baseline (Table 4)
  lm_eval_adapter.py          optional drop-in model for EleutherAI's LM Evaluation Harness
  assistant_prompts.py        Appendix G prompts (Section 3.4, out of scope, mechanism only)

experiments/                  one script per paper experiment (all have --help)
  run_zeroshot.py             Section 3.1 / Table 5
  run_humaneval.py            Section 3.3.1 / Table 2
  run_cot.py                  Section 3.2 (+ self-consistency)
  run_flops_ancova.py         Section 4 / Table 6 / Figure 9
  run_section5.py             Sections 5.1-5.2
  run_visualize_logits.py     Section 5.3 / Table 3
  run_fudge_comparison.py     Section 6 / Table 4
  run_assistant_negative_prompting.py   Section 3.4 mechanism (out of scope)
  plot_figures.py             Figures 2/6/7/17/18 from the JSON results
  smoke_test.py               end-to-end pipeline check without any downloads

scripts/                      shell entry points (run_tests.sh runs offline in seconds)
configs/                      the sweeps in JSON (models, gammas, temperatures, k)
tests/                        54 offline unit/integration tests
```

---

## 3. Scope (following `paper/addendum.md`)

| Paper item | In scope? | Where |
|---|---|---|
| Eq. 5/7 CFG formulation + negative prompting | yes | `cfglm/cfg.py`, `cfglm/generation.py` |
| 3.1 zero-shot benchmarks, **Table 5**, Figs 6-8 | yes (Table 5; LLaMA excluded) | `experiments/run_zeroshot.py` |
| 3.2 Chain-of-Thought, Figs 2/17 | yes | `experiments/run_cot.py` |
| 3.3.1 HumanEval, **Table 2**, Fig 3, Tables 7-9 | yes | `experiments/run_humaneval.py` |
| 3.4 negative prompting / human preference study | no (human study) | mechanism only in `assistant_prompts.py` |
| 4.1 FLOPs experiment, **Table 6** | yes | `experiments/run_flops_ancova.py` |
| 4 VRAM/memory analysis (analytic) | no | `flops.kv_cache_bytes` provided for reference |
| 5.1 entropy analysis | yes | `experiments/run_section5.py` |
| 5.2 CFG vs instruction tuning, Fig 5, Tables 12-14 | yes | `experiments/run_section5.py` |
| 5.3 vocabulary visualisation, **Table 3** | yes | `experiments/run_visualize_logits.py` |
| 6 **Table 4** (CFG vs FUDGE) | yes | `experiments/run_fudge_comparison.py` |
| **Table 1** (assistant demo) | no | - |
| Figs 6, 7, 9, 11, 12, 13 (charts) | excluded as charts | the underlying data is still produced |
| Appendix D.1 GPT-J / CodeGen probing; LLaMA models | no | - |

Models used: GPT-2 (s/m/l/xl), Pythia (160M-12B), CodeGen-mono (350M/2B/6B),
WizardLM-30B, Guanaco-65B, Falcon-7b (base + instruct).  All are open weights
on the HuggingFace Hub, so **no third-party API keys are required** (the
provided `agent.env` is empty and unused).

---

## 4. What has been verified in this environment

This machine has **no GPU and no network access**, so the GPU experiments
cannot be executed here.  What *was* executed:

* `python -m pytest tests -q` -> **54 passed** (see `tests/`).  These run
  against tiny, randomly-initialised GPT-2 models and synthetic data and
  verify, among other things:
  * the equivalence of the logit-space and log-probability-space forms of
    Eq. 7, and that `gamma = 1` is exactly the conditional distribution;
  * that `CFGScorer` reproduces an *independently written* implementation of
    Eq. 7 token by token, and that its batched path equals its unbatched path;
  * that `gamma = 1` generation is token-for-token identical to vanilla
    greedy decoding;
  * that a negative prompt (Eq. 5) changes the unconditional branch;
  * that CFG doubles the FLOP estimate;
  * the pass@k estimator, the entropy/overlap statistics and the ANCOVA
    (recovering a known group effect, and finding none when there is none);
  * the prompt formats of the nine benchmarks, the CoT answer parser, and the
    HumanEval truncation / execution / win-tie-loss paths.
* `python experiments/smoke_test.py` -> runs the complete sweep +
  FLOP/ANCOVA pipeline end to end on a synthetic benchmark.
* `bash scripts/run_tests.sh` -> both of the above.

Everything else (`scripts/run_all.sh`) is written for the GPU environment in
which the paper's numbers were produced; the scripts print progress and write
JSON/PNG artefacts under `results/`.

---

## 5. How to reproduce the paper's results

```bash
pip install -r requirements.txt

bash scripts/run_tests.sh        # seconds, CPU, no downloads
bash scripts/run_all.sh          # full reproduction, needs a GPU box
```

Individual experiments:

```bash
bash scripts/run_zeroshot.sh     # Table 5 / Figures 6-8 data
bash scripts/run_flops_ancova.sh # Table 6 / Figure 9
bash scripts/run_humaneval.sh    # Table 2, Tables 7-9, Figure 3
bash scripts/run_cot.sh          # Figures 2 and 17
bash scripts/run_section5.sh     # Sections 5.1-5.3
bash scripts/run_table4_fudge.sh # Table 4
```

The zero-shot table can alternatively be produced with the *canonical*
EleutherAI LM Evaluation Harness (which is what the paper used):

```bash
python experiments/run_zeroshot.py --backend lm_eval --models gpt2-xl \
    --gammas 1.0 1.25 1.5 --output results/zeroshot_lm_eval.json

# ... or directly through the adapter
python -m cfglm.lm_eval_adapter --model gpt2-xl --gamma 1.5 \
    --tasks arc_challenge,arc_easy,boolq,hellaswag,piqa,sciq,triviaqa,winogrande,lambada_openai
```

`cfglm.lm_eval_adapter.CFGHFLM` subclasses the harness's `HFLM` and replaces
its likelihood routines with the CFG scoring, so the harness's task
definitions, metrics and few-shot formatting are untouched.  Both backends
write the same JSON schema, so the FLOP/ANCOVA analysis works with either.

---

## 6. Results to expect (the paper's trends)

These are the paper's reported values, which the scripts are written to
produce; a reproduction should match the trends within a reasonable margin.

* **Table 5 / Section 3.1** - CFG improves most zero-shot benchmarks, with
  ARC-challenge and WinoGrande the two exceptions.  LAMBADA improves
  substantially: GPT-2 small `32.6 -> 44.6` at `gamma = 1.5`, Pythia-2.8B
  `64.6 -> 76.5`, and LLaMA-7B reaches `81.3`, above the zero-shot
  PaLM-540B SOTA of `77.9` (LLaMA is out of scope here).
* **Table 2 / Section 3.3.1** - CodeGen pass@1 rises for `1 <= gamma <= 1.5`
  (350M: `11.0% -> 11.8%` at `gamma = 1.1`) and falls for larger gamma, while
  pass@100 *decreases* (350M: `22.0% -> 18-20%`), i.e. CFG sharpens the model
  at a small cost in diversity.  Figure 3 counts more tasks where CFG wins
  than loses at `gamma = 1.25`.
* **Figures 2 / 17 / Section 3.2** - for small gamma, CFG increases both
  accuracy and the fraction of *valid* (parsable) reasoning chains; beyond
  `gamma = 1.5` chains degrade and accuracy drops.
* **Table 6 / Section 4.1** - across the nine tasks, 5 of 9 show a
  statistically insignificant difference (`p > .01`) between
  "small model + CFG" and "twice-as-large model, vanilla" when both are
  plotted against inference FLOPs per token; of the significant ones, LAMBADA
  and SciQ favour CFG while WinoGrande and TriviaQA favour vanilla.
* **Section 5.1** - CFG lowers the mean sampling entropy (the paper reports
  `4.7` vs `5.49` nats for the prompted distribution), and CFG shares roughly
  `50%` of the top-p = 90% tokens with the vanilla prompted model.
* **Section 5.2** - CFG and instruction tuning have similar entropy but
  largely non-overlapping vocabularies; where they do agree, the Spearman
  correlation between the two models' continuation perplexities exceeds
  `r_s > .7`, especially for longer, more specific prompts.  The paper's
  perplexity correlation matrix is `0.94` (prompted-CFG), `0.83`
  (prompted-instruct) and `0.70` (CFG-instruct).
* **Table 3 / Section 5.3** - with `c = "The dragon flew over Paris, France"`,
  guided sampling up-weights tokens about dragons and Paris and down-weights
  other locations ("Queensland"), dates ("1913") and topics ("hostages",
  "voyages").
* **Table 4 / Section 6** - CFG steers external classifiers more than FUDGE
  (`0.312` vs `0.065` for sentiment, `0.523` vs `0.045` for toxicity) at a
  fraction of the cost: CFG needs one extra forward pass, whereas FUDGE needs
  a discriminator call at *every* timestep.

---

## 7. Implementation notes and deliberate choices

* **Where the unconditional prompt starts.**  Section 3.1 says CFG is
  implemented "by starting the unconditional prompt at the last token of the
  initial prompt".  `uncond_prefix_tokens = 1` (the default) does exactly
  that, and every experiment exposes the flag.
* **Scoring vs sampling.**  Likelihood-based benchmarks use per-token CFG
  *log-probabilities* summed over the continuation (`CFGScorer`), which is
  the sequence-level form of Eq. 7 (`proportional to P(w|c)^gamma /
  P(w)^(gamma-1)`).  Generative experiments (HumanEval, CoT, assistants) use
  the logits processor, so temperature and top-p are applied by the standard
  sampler *after* the CFG combination.
* **`is_greedy`.**  LAMBADA's accuracy needs the argmax of the *CFG*
  distribution; the scorer therefore rebuilds the CFG logits instead of
  reusing the conditional argmax.
* **Task definitions.**  `cfglm/tasks.py` mirrors the harness's zero-shot
  formatting (ARC / BoolQ / PIQA / SciQ use `Question: ...\nAnswer:` with a
  space-prefixed continuation; HellaSwag, LAMBADA and WinoGrande concatenate
  without one; WinoGrande scores the option at the blank; TriviaQA uses the
  LLaMA-style methodology with **substring** match, per the addendum).
  Installing `lm-eval` and using `cfglm/lm_eval_adapter.py` switches to the
  canonical definitions for the exact numbers.
* **`acc_norm`** is normalised by the number of *continuation tokens*, which
  follows the harness convention; `acc` is the metric quoted in Table 5.
* **FLOPs.**  `cfglm/flops.py` follows `electra/flops_computation.py`,
  counting a MAC as 2 FLOPs, and includes the `4*S*d` per-layer attention
  term so that cost grows with context length.  CFG uses `n_passes = 2`.
  The per-task sequence lengths used for the attention term are listed in
  `TASK_SEQ_LEN` inside `run_zeroshot.py` and can be adjusted.
* **ANCOVA.**  `cfglm/stats.py:ancova` fits
  `accuracy ~ log(FLOPs/token) + group` and reports the group p-value (the
  adjusted-means test of Table 6); an interaction (slope-difference) F-test is
  also returned.  Significance is read at `p = .01`.
* **P3 sample.**  `cfglm/p3.py` enumerates `bigscience/P3` configurations,
  takes up to `n_per_dataset = 50` documents from each, and drops documents
  whose input exceeds 200 tokens, matching the addendum.  Because the Hub
  revision changes over time, the script reports the number of datapoints it
  actually collected so it can be compared with the paper's 32,902 (the
  addendum's "about 50 x 660 datasets").
* **FUDGE approximation.**  `cfglm/fudge.py` reweights the top-k candidates at
  each step with the external classifier's score of the partial continuation.
  This keeps FUDGE's defining property (an external-classifier call at every
  timestep, and the resulting cost) and its qualitative conclusion, without
  reproducing the original paper's exact future-rollout marginalisation.
  This is the one place where a *baseline* - not a paper contribution - is
  approximated, and it is flagged in the docstring.
* **HumanEval execution.**  Generated programs run in a fresh subprocess with
  a timeout (`cfglm/humaneval.py`), so infinite loops or `sys.exit` cannot
  hang the sweep; completions are truncated at the standard CodeGen stops
  before the unit tests are appended.  The default ``--execution inline``
  mode mirrors OpenAI's ``human-eval`` runner (in-process ``exec`` guarded by
  ``SIGALRM``), which is needed for the sweep to be practical; the isolated
  ``--execution subprocess`` mode is also available, and both are tested.

---

## 8. Compute expectations

| Experiment | Rough cost on one A100-80G |
|---|---|
| Table 5 sweep (11 models x 9 tasks x 6 gammas) | ~2-4 GPU-days (dominated by HellaSwag/BoolQ) |
| HumanEval (3 CodeGen x 3 temps x 6 gammas x 100 samples) | ~3-5 GPU-days |
| CoT (2 models of 30-65B, GSM8K + AQuA) | ~2 GPU-days (needs 2 x 80 GB for 65B) |
| Section 5 (Falcon-7b x 32,902 datapoints x 4 distributions) | ~1 GPU-day |
| Table 4 (GPT-2, 64 samples x sweeps) | less than 1 GPU-hour |

The scripts write partial results after every configuration, so a sweep can be
interrupted and resumed by re-running with the same `--output`.
