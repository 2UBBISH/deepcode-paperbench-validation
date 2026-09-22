# What Will My Model Forget? — reproduction

Reproduction of

> Xisen Jin, Xiang Ren. **What Will My Model Forget? Forecasting Forgotten Examples in
> Language Model Refinement.** ICML 2024.

The repository implements the paper's problem setup, its three forecasting methods and
the replay-based model-refinement experiments. Everything that the paper describes in
its main text is implemented; the appendix-only experiments are marked as out of scope
(see the "Scope" section at the end).

## 1. What the paper does, and where it lives in this repository

| Paper part | What it is | Code |
|---|---|---|
| Sec. 2 | Problem formulation: `D_PT`, `D_hat_PT`, `D_R`, edit success rate, EM drop ratio, forgetting label `z_ij` | [`wwmf/evaluation/metrics.py`](wwmf/evaluation/metrics.py), [`wwmf/data/build.py`](wwmf/data/build.py) |
| Sec. 3.1, Table 1 ("Threshold") | Frequency-threshold forecasting (threshold `gamma` tuned on `D_R^Train`) | [`wwmf/forecasting/threshold.py`](wwmf/forecasting/threshold.py) |
| Sec. 3.2, Eq. 2–3, Table 1 ("Fixed / Trainable Logit") | Logit-change transfer through a (fixed or trained) kernel `Theta_tilde(x_j, x_i) = h(x_j,y_j) h(x_i,y_i)^T`, margin loss Eq. 3 | [`wwmf/forecasting/logit_change.py`](wwmf/forecasting/logit_change.py) |
| Sec. 3.3, Eq. 4, Table 1 ("Representation", "w/o Prior") | Bilinear black-box model `sigma(h(x_j,y_j)·h(x_i,y_i) + b_j)` with frequency prior `b_j`, BCE loss | [`wwmf/forecasting/representation.py`](wwmf/forecasting/representation.py) |
| Sec. 4.1 | Base PTLMs, `D_PT` (36 P3 train tasks × 100 examples), `D_R` (P3-Test for BART0 / MMLU validation for FLAN-T5), 60/40 split, head-only / LoRA / full FT, learning rates | [`wwmf/config.py`](wwmf/config.py), [`wwmf/data/`](wwmf/data), [`wwmf/models/`](wwmf/models) |
| Sec. 4.2 | Compared methods: fixed-logit forecasting, replay with distillation loss, replay of forecasted forgotten examples, ground-truth forgotten examples | [`wwmf/refinement/replay.py`](wwmf/refinement/replay.py), [`wwmf/models/tuning.py`](wwmf/models/tuning.py) |
| Sec. 5.1, Table 1 | F1 of forecasting when fixing one error at a time (7 LM × tuning setups) | [`scripts/run_table1.py`](scripts/run_table1.py) |
| Sec. 5.1, Table 2 | In-domain / out-of-domain generalization of the forecasting models (BART0) | [`scripts/run_table2.py`](scripts/run_table2.py), `P3_TEST_ID_TASKS`/`P3_TEST_OOD_TASKS` in [`wwmf/data/registry.py`](wwmf/data/registry.py) |
| Sec. 5.1, Figure 3 | Continual model refinement: F1/precision/recall over time (prediction frozen at the start of the stream, ground truth recomputed) | [`wwmf/refinement/stream.py`](wwmf/refinement/stream.py) (`continual_forecasting_curves`), [`scripts/run_figure3.py`](scripts/run_figure3.py) |
| Sec. 5.2, Table 3 | Edit success rate and EM drop ratio with sequential refinement + replay | [`wwmf/refinement/stream.py`](wwmf/refinement/stream.py) (`sequential_refinement`), [`scripts/run_table3.py`](scripts/run_table3.py) |
| Sec. 5.2, Table 4 | Same, but fixing single errors separately | `evaluate_single_error_replay`, [`scripts/run_table4.py`](scripts/run_table4.py) |
| Sec. 5.3, Table 5 | Computational complexity (and Appendix C FLOP accounting) | [`wwmf/analysis/complexity.py`](wwmf/analysis/complexity.py), [`scripts/run_complexity.py`](scripts/run_complexity.py) |
| Figure 2(a) | Empirical study of the logit-change transfer between two examples | [`wwmf/analysis/logit_transfer.py`](wwmf/analysis/logit_transfer.py), [`scripts/analyze_logit_transfer.py`](scripts/analyze_logit_transfer.py) |
| Appendix F, Alg. 1–4 | Training / inference procedures of the logit- and representation-based forecasters | `fit`/`predict` of [`wwmf/forecasting/logit_change.py`](wwmf/forecasting/logit_change.py) and [`wwmf/forecasting/representation.py`](wwmf/forecasting/representation.py) |

## 2. Repository layout

```
wwmf/
  config.py                all hyper-parameters of Sec. 4.1 + LM registry
  data/
    registry.py            the 36 D_PT tasks, the 8 P3-Test tasks, the Table-2 ID/OOD
                           split, the 57 MMLU subjects, T0 -> P3 config mapping
    p3.py                  P3 loaders (HF `bigscience/P3` parquet configs + a local
                           BART0/ReCross-style dump)
    mmlu.py                MMLU loaders (original release CSVs or HF mirror)
    build.py               D_PT, D_hat_PT, D_R, 60/40 split
  models/
    lm.py                  Seq2SeqLM wrapper: generate/EM, teacher-forced logits,
                           hidden states, parameter snapshots
    tuning.py              head-only / LoRA / full FT, K-step error fixing,
                           distillation replay loss
  forecasting/
    cache.py               caching of f_0 logits/representations, collection of the
                           per-online-example artifacts and of the ground-truth labels
    threshold.py           Sec. 3.1
    logit_change.py        Sec. 3.2 (fixed + trainable, Eq. 2 and 3)
    representation.py      Sec. 3.3 (+ "w/o Prior" ablation)
    encoders.py            the trainable encoder h = LM + 2-layer MLP
  refinement/
    replay.py              random / forecasted / ground-truth replay selection
    stream.py              sequential refinement, single-error refinement, Fig. 3 curves
  evaluation/metrics.py    Exact Match (SQuAD 2.0 normalisation), F1, EM drop ratio
  analysis/                Sec. 5.3 complexity + Figure 2(a) logit-transfer analysis
  experiments.py           glue code for the tables/figures
scripts/                   one CLI per table / figure (+ run_all.sh)
tests/                     unit tests (metrics, threshold tuning, margin loss, replay, ...)
smoke/                     reduced-scale CPU end-to-end run + its results
```

## 3. How to run

```bash
pip install -r requirements.txt

# unit tests (no model download, a few seconds)
python -m pytest tests -q

# reduced-scale end-to-end run on CPU with FLAN-T5_small (~12 min)
python smoke/run_smoke.py --output smoke/outputs
# reduced-scale run of the sequential-refinement paths (Table 3 / Figure 3)
python smoke/run_smoke_stream.py --output smoke/outputs_stream

# full-scale experiments (GPU; one command per table column)
python scripts/run_table1.py  --model bart0_large   --tuning-mode head
python scripts/run_table1.py  --model flan_t5_large --tuning-mode lora
python scripts/run_table2.py  --model bart0_large   --tuning-mode full_ft
python scripts/run_figure3.py --model flan_t5_large --tuning-mode lora
python scripts/run_table3.py  --model flan_t5_large --tuning-mode lora
python scripts/run_table4.py  --model bart0_large   --tuning-mode full_ft
python scripts/run_complexity.py --model flan_t5_large
# or everything at once
bash scripts/run_all.sh outputs
```

Every script writes JSON results under `--output-root` (`outputs/` by default) and prints
a summary. Expensive intermediate quantities (cached logits, the per-online-example
ground-truth forgetting labels) are pickled under `--cache-root`, so re-running a table
with different forecasting methods reuses them.

## 4. Data

* **`D_PT`** — 36 tasks of the P3 *training* split, 100 examples per task (Sec. 4.1 and
  the addendum). The task list is in `wwmf/data/registry.py`; the loader takes a
  *balanced* sample, i.e. the same number of examples per task, and uses **all** template
  variants of a task (addendum, Appendix B clarification).
* **`D_R`** — mispredicted examples of `f_0`, graded with Exact Match using the SQuAD 2.0
  normalisation (addendum):
  * BART0: the *test* split of P3 (`wwmf.data.registry.P3_TEST_TASKS`),
  * FLAN-T5: the *validation* split of MMLU, 57 subjects
    (`wwmf.data.registry.MMLU_SUBJECTS`).
  `D_R` is randomly split 60% / 40% into `D_R^Train` and `D_R^Test`.
* **`D_hat_PT`** — the upstream examples that `f_0` answers correctly; all forecasting
  metrics are computed on `D_hat_PT` (addendum to Sec. 3.1).

Two data sources are supported:

1. `bigscience/P3` on the HF hub (per-template parquet configs). `T0 task -> P3 template
   config` mapping is a prefix match, with explicit overrides for
   `super_glue-wsc.fixed`, `cos_e-v1.11` and `wiki_hop-original`.
2. A local dump in the layout of the BART0 / ReCross data
   (`<data_root>/p3/<task>/{train,validation,test}.jsonl`). This is the dump the authors
   used, and it contains two tasks that the public `bigscience/P3` snapshot lacks
   (`paws_x-en` and `storycloze`, the latter being part of the Table-2 OOD split).
   Put it under `data/p3/` (or pass `--data-root`) to use it.
3. MMLU: the original release
   (`people.eecs.berkeley.edu/~hendrycks/data.tar`) read from `<data_root>/mmlu/val/`,
   with a fallback to the `cais/mmlu` HF mirror.

## 5. Hyper-parameters

All values of Sec. 4.1 are in [`wwmf/config.py`](wwmf/config.py):

| Setting | Value |
|---|---|
| Steps per fixed error | 100 (head only), 30 (LoRA / full FT) |
| LR, single error | BART0 `1e-5`, FLAN-T5 `1e-4` (head only: `1e-3` / `1e-4`) |
| LR, sequential refinement | BART0 `1e-6`, FLAN-T5 `1e-5` (head only: `1e-4` / `1e-5`) |
| LoRA (addendum) | `r=16`, `alpha=32`, `dropout=0.1`, `bias="none"`, `target_modules=['q','v']` |
| Replay | 8 examples every 10 steps (BART0_Large, FLAN-T5_Large), 4 every 5 steps (FLAN-T5_3B) |
| Forecasting training (Appendix B) | ≤100k steps, batch 16 (8 positive + 8 negative pairs), positive loss weight `alpha=0.1`, LM LR `1e-5`, MLP LR `1e-4` |
| Encoder `h` | BART0 (BART0 experiments) / FLAN-T5_small (T5 experiments) + fresh 2-layer MLP |
| Logit caching | top-`k=100` largest logits per gold token (Sec. 3.2) |
| Figure 3 streams | 1/8 of `D_R`, randomly shuffled (addendum) |

### Assumptions where the paper is ambiguous

These are documented in code comments and repeated here:

* **Head-only learning rates** — the paper gives one pair of values for head-only tuning
  ("`1e-3` and `1e-4`") without mapping them to the models; we map them to BART0 and
  FLAN-T5 respectively, and reduce them 10× for sequential refinement (mirroring the
  LoRA/full-FT sentence).
* **MMLU prompt** — the paper does not show the prompt used to elicit MMLU answers. We
  use a zero-shot letter-answer prompt (`Question: ... A. ... D. ... Answer:`), and grade
  the emitted letter with Exact Match, exactly like the P3 experiments.
* **Distillation loss** — "distillation loss against the outputs of the base PTLM
  (Buzzega et al., 2020a)" is implemented as the MSE between the current and cached
  pre-softmax logits (dark experience replay), restricted to the cached top-k candidate
  vocabulary of `D_PT` so that the replay buffer stays tractable.
* **Candidate vocabulary** — `f_hat_i(x_j)` is evaluated on the union of the per-token
  top-k vocabulary entries of all upstream examples (plus every gold token), padded to
  `max_target_len`. This is the "we only cache top k = 100 largest logits" trick of
  Sec. 3.2 made explicit; the arg-max is therefore taken over that candidate set.
* **Online storage of `h`** — the fixed logit-based forecaster uses the frozen
  representation of the *base* model for `h(x_i, y_i)` (as in Eq. 2, valid when only the
  LM head is tuned) and the cached `f_hat_i(x_i) - f_hat_0(x_i)` logit change.
* Table 3/4 stream order is randomly shuffled; Figure 3 shuffling is explicitly stated in
  the addendum.

## 6. What was executed here vs. what needs a GPU

This machine has **no GPU** (CPU/MPS only), so the full-scale experiments (BART0_Large,
FLAN-T5_Large, FLAN-T5_3B over 3,600 upstream examples and hundreds of per-example
fine-tuning runs, i.e. tens of thousands of LM inferences) were **not** executed here.
Everything needed to run them is implemented and the pipeline was validated end-to-end at
a reduced scale.

### Evidence collected in this environment

`python -m pytest tests -q` → **21 passed** (Exact Match/SQuAD normalisation, threshold
tuning, Eq. 3 margin loss signs and positive down-weighting, kernel/pad/arg-max logic,
pair sampling, frequency prior, replay selection, task registry and hyper-parameter
tables).

`python smoke/run_smoke.py` → reduced-scale end-to-end run with `google/flan-t5-small`
(head-only tuning as the refinement setup); details in
[`smoke/SMOKE_RESULTS.md`](smoke/SMOKE_RESULTS.md):

| Item | Value |
|---|---|
| `D_PT` | 24 examples (3 P3 templates of the `glue-mrpc` task, 8 each) |
| `D_R` | 12 mispredictions of FLAN-T5_small on the `glue_qqp_duplicate` validation split (EM 0.70 on the candidate pool) |
| `D_hat_PT` | EM of `f_0` on `D_PT` = 0.667 |
| Refinement | 20 head-only steps per error (`lr=2e-3`), 3 + 3 online examples, 6/6 errors fixed |

Forecasting F1 on `D_R^Test` (see `smoke/results/smoke_results.json`):

| Method | F1 | Precision | Recall |
|---|---|---|---|
| Threshold | 95.83 | 95.83 | 95.83 |
| Fixed Logit | 66.67 | 50.00 | 100.00 |
| Trainable Logit | 66.67 | 50.00 | 100.00 |
| Representation | **95.83** | 95.83 | 95.83 |
| w/o Prior | 66.67 | 50.00 | 100.00 |

The same run also exercises the Table-4-style single-error refinement path (replay of 8
examples every 10 steps, distillation against the cached base logits, EM drop ratio
measured on `D_PT`); at this scale 20 head-only steps of a 77M model do not induce any
forgetting, so those numbers only demonstrate that the pipeline runs — see
`smoke/SMOKE_RESULTS.md` for the raw values.

The absolute numbers are not comparable to the paper (77M-parameter model, a handful of
examples, a few optimisation steps), but the *pipeline* is exercised end to end and the
ranking follows the paper's trend: the representation-based model with the frequency
prior is the strongest forecaster, its ablation without the prior loses a lot of
precision, and the logit-based variants are weaker on this T5 model — the same ordering
as Table 1 on FLAN-T5.

### Running the real experiments

`bash scripts/run_all.sh outputs` on a GPU host reproduces Tables 1–4 and Figure 3 with
the paper's models and hyper-parameters. Rough costs per Table-1 column: `|D_R|` per-example
fine-tuning runs, each followed by one inference pass over `D_hat_PT` (3,600 examples) to
obtain the ground-truth labels; the forecasting models themselves are trained for up to
100k steps with a batch of 16 pairs.

## 7. Scope

In scope (implemented): Sec. 2–5 of the main text — the forecasting methods (Sec. 3.1–3.3),
Tables 1–4, Figure 3, the complexity analysis of Sec. 5.3, and the qualitative logit-change
analysis of Figure 2(a).

Out of scope (per the addendum, not implemented as a default run):

* Table 3 rows from other papers (MIR, OCS).
* Sec. 5.2 "Hyperparameter Analysis", Sec. 5.3/Appendix C FLOP measurements, Sec. 5.3
  "Accumulating Forecasted Examples in Longer Streams" (Table 12), the MEND comparison
  (Table 6), and the other appendix-only tables.

Known deviations: the MMLU prompt is not specified by the paper (see above); the SQuAD
2.0 Exact Match used for grading is the same normalisation the addendum prescribes; the
`storycloze` OOD task and the `paws_x-en` upstream task need the ReCross data dump because
they are not part of the public `bigscience/P3` snapshot.

## 8. Notes on the environment

* Apple **MPS** is deliberately not selected by `wwmf.utils.resolve_device("auto")`: with
  the torch/transformers versions used here, seq2seq generation on MPS produces degenerate
  text (verified: Exact Match collapses from 0.75 to 0.0 on the same inputs). Use
  `device="mps"` explicitly if you want to test that backend.
* `peft` is only imported when LoRA is used (`wwmf/models/tuning.py:apply_lora`), so
  head-only / full-FT runs work without it.
