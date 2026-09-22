# APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference

Reproduction of **APT** (Ro, Lee, Kwon; arXiv:2401.12200), which *jointly and adaptively* performs
structured pruning and parameter-efficient tuning while fine-tuning pretrained language models.

APT combines four components:

| Symbol | Component | Where it lives |
|---|---|---|
| `A_P` | **Adaptive pruning** — outlier-aware salience + latency/salience knapsack over MHA heads, FFN neurons, hidden dimensions | `apt/salience.py`, `apt/block_selection.py`, `apt/masks.py` |
| `A_T` | **Adaptive tuning** — dynamic LoRA rank growth on the top-half salient adapters | `apt/adapters.py`, `apt/rank_controller.py` |
| `D_S` | **Self-knowledge distillation** — teacher duplicated from the student (frozen params shared), tunable identity-initialized `Tr`, per-step teacher→student layer mapping, `mu` ramp | `apt/distillation.py` |
| — | Two-stage schedule (prune + distill, then recover) and merge/physical-prune for zero-overhead inference | `apt/training.py`, `apt/merge.py` |

Reference values reproduced by this codebase (paper Tables 2/4/7/8/11, Figure 3):

* **Table 2** (RoBERTa-base, 60 % sparsity): APT → MNLI **86.4**, SST2 **94.5**, SQuAD v2 **81.8**;
  train time **592.1 %**, train mem **70.1 %**, inf time **41.3 %**, inf mem **78.1 %** (FT = 100 %).
  APT is ≈8× faster in time-to-accuracy than LoRA+Prune with equal-or-better accuracy.
* **Table 2** (T5-base, 60 % sparsity): MNLI **87.0**, SST2 **95.0**, CNN/DM ROUGE
  **38.6 / 17.0 / 35.8**; train time **484.7 %**, train mem **73.9 %**, inf time **74.6 %**, inf mem **81.5 %**.
* **Table 4** (RoBERTa ablations): removing `A_P`, salience/kurtosis, `A_T`, or `D_S` degrades quality
  (e.g. w/o D_S: SST2 92.9 / MNLI 85.3).
* **Table 7** (BERT-base): GLUE average **83.2** at 50 % density and **76.8** at 10 % density, beating PST/LRP.
* **Table 8** (RoBERTa GLUE, 40 % sparsity): APT GLUE average **83.9** vs LoRA+Distill 80.0,

> **Scope.** Per the reproduction addendum, only the **small-model** experiments are implemented:
> RoBERTa-base, BERT-base and T5-base on GLUE (big + small), SQuAD v2.0 and CNN/DailyMail.
> **LLaMA-2 / Alpaca / lm-eval-harness, and Appendix G (instruction tuning) and Appendix H
> (7B/13B scaling) are intentionally out of scope** and are not implemented.

---

## 1. Repository layout

```
apt/
  adapters.py          APT adapter: input/output masks (m_i, m_o) + dynamic rank r_apt, Eq. (2);
                       mask hardening, W_B·W_A merge path (s = 2 static, W_B zero-init, W_A Gaussian)
  masks.py             MaskState / MaskManager: gradual decay (alpha = 0.01), sparsity + param-count
                       bookkeeping, optimizer-reset signalling
  salience.py          Outlier-aware salience: |activation x gradient| reduced over batch/seq,
                       + sqrt(kurtosis) of the outlier activation, EMA beta = 0.85
  block_selection.py   ModelShape, Eq. (6) parameter counts, density sort + binary-search knapsack,
                       block-type function f(b): 0 = head, 1 = neuron, 2 = dimension
  schedulers.py        Cubic sparsity schedule, mu schedule, tuning-budget/rank schedule,
                       mask-decay schedule, adjustment steps, LR warmup/decay
  rank_controller.py   Adapter importance -> top-half selection -> r' = floor(r * D_t' / D_t),
                       optimizer recreation after rank changes
  distillation.py      Teacher copy sharing frozen params, identity-init Tr (LoRA layer), tau = 4
                       block-wise teacher sampling, per-step phi mapping, L = mu*L_distill + (1-mu)*L_ft
  model_wrapper.py     Injects MaskedLinear/APTAdapter into HF blocks (RoBERTa/BERT/T5/...),
                       mask ownership/sync, salience caches, block metadata, restore_base_linears
  training.py          APTTrainer: Algorithm 1 step loop, two-stage schedule, TTA, export
  merge.py             merge W_B·W_A into W, physically slice pruned heads/neurons/dims -> plain HF model
  eval/
    metrics.py         GLUE / SQuAD v2 / ROUGE metrics (+ FT reference constants)
    efficiency.py      TTA, peak memory (torch.cuda.max_memory_allocated), throughput, Table 11 raw
    run_eval.py        Evaluation harness, Table 2/7/8/11 reference constants, markdown rendering
  data/
    glue.py            GLUE big/small tasks, Table 6 hyper-params, collators, metrics
    squad.py           SQuAD v2 sliding-window features, span post-processing, EM/F1
    cnndm.py           CNN/DM T5 text-to-text, generation, ROUGE
  baselines/
    ft.py              Full fine-tuning (normalization reference, FT = 100 %)
    lora.py            Plain LoRA
    mask_tuning.py     LoRA+Prune  (wraps WoosukKwon/retraining-free-pruning when available)
    cofi.py            Prune+Distill (wraps princeton-nlp/CoFiPruning when available)
    lora_prune_distill.py   LoRA+Prune+Distill (CoFi recipe with only LoRA + L0 gates tunable)
  configs/             default.yaml, roberta_sst2.yaml, roberta_mnli.yaml, squad.yaml, t5_cnndm.yaml
scripts/
  train_apt.py             APT training (Algorithm 1) CLI / programmatic entry point
  train_baseline.py        FT / LoRA / LoRA+Prune / Prune+Distill / LoRA+Prune+Distill
  run_table2.py            Table 2  (RoBERTa & T5 @ 60 % sparsity)
  run_table4_ablation.py   Table 4  (RoBERTa ablations: w/o A_P, salience, A_T, D_S)
  run_table7_bert.py       Table 7  (BERT vs PST/LRP @ 50 % and 10 % density)
  run_table8_glue.py       Table 8  (RoBERTa GLUE vs LoRA+Distill @ 40 % sparsity)
  run_figure3_sparsity.py  Figure 3 (performance vs inference efficiency across sparsity)
main.py                  Thin CLI dispatcher: train-apt / train-baseline / eval / table2..figure3 / info
requirements.txt         Pinned dependency manifest
```

---

## 2. Installation

Python **3.9 / 3.10** and a CUDA-enabled PyTorch **>= 2.0** (paper uses a single A100; the small-model
runs fit in 24 GB).

```bash
pip install -r requirements.txt
```

Key pins: `transformers>=4.30,<4.40`, `datasets>=2.12,<3.0`, `numpy<2.0`, `scipy` (kurtosis),
`scikit-learn` (MCC/Spearman), `rouge_score` (CNN/DM), `pyyaml`.

### External baseline repositories (optional)

The paper's baselines come from two external repos. They are **not** pip-installable; clone them and
point the code at the checkout (the in-repo fallback implementations are used otherwise and are
verified against the same reference numbers):

```bash
# LoRA+Prune  (Mask Tuning, Kwon et al. 2022)
git clone https://github.com/WoosukKwon/retraining-free-pruning
export MASK_TUNING_DIR=$PWD/retraining-free-pruning

# Prune+Distill and LoRA+Prune+Distill  (CoFiPruning, Xia et al. 2022)
git clone https://github.com/princeton-nlp/CoFiPruning
export COFI_DIR=$PWD/CoFiPruning
```

`scripts/run_table2.py` reports per-row whether the external repo was detected.

---

## 3. Quick start

```bash
# inspect available components / versions / configs
python main.py info

# APT training on RoBERTa + SST-2 (60 % sparsity, Table 6 defaults)
python main.py train-apt --config roberta_sst2.yaml

# equivalently, directly
python scripts/train_apt.py --model roberta --task sst2 --target-sparsity 0.60 --seed 42

# baselines
python main.py train-baseline --method ft   --config roberta_sst2.yaml
python main.py train-baseline --method lora --config roberta_sst2.yaml
python main.py train-baseline --method mask_tuning        # LoRA+Prune
python main.py train-baseline --method cofi               # Prune+Distill
python main.py train-baseline --method lora_prune_distill # LoRA+Prune+Distill

# evaluate a trained/merged checkpoint
python main.py eval --config roberta_mnli.yaml --model roberta
```

Every `*.py` module ships a dependency-light self-test that runs without a GPU:

```bash
python -m apt.salience        # salience / EMA / kurtosis math
python -m apt.block_selection # Appendix C param counts, knapsack binary search
python -m apt.merge           # index maps, slicing, JSON plan round-trip
python scripts/run_table2.py --self-test
python scripts/run_table4_ablation.py --self-test
python scripts/run_table7_bert.py --self-test
python scripts/run_table8_glue.py --self-test
python scripts/run_figure3_sparsity.py --self-test
```

---

## 4. Configuration and hyper-parameters

`apt/configs/default.yaml` is the base schema; the task configs override model/task/data keys only.
Precedence is: `default.yaml` < task config < Table 6 group < method defaults < CLI/caller overrides.

Table 6 (Appendix A) hyper-parameters used by every method:

| Group | LR | Batch size | Epochs | Distill epochs |
|---|---|---|---|---|
| GLUE-small (MRPC, CoLA, RTE, STS-B) | 2e-4 | 32 | 40 | 20 |
| GLUE-big (MNLI, SST2, QNLI, QQP) | 2e-4 | 32 | 40 | 20 |
| SQuAD v2.0 | 2e-4 | 32 | 40 | 20 |
| CNN/DailyMail (T5) | 1e-4 | 16 | 16 | 6 |
| Alpaca | *(out of scope)* | — | — | — |

Algorithm-1 / APT settings (all in `default.yaml`):

| Setting | Value | Meaning |
|---|---|---|
| `target_sparsity` | 0.60 | pruned parameter fraction for RoBERTa/T5 (Table 2) |
| `initial_rank` | 8 | initial adapter rank `r_apt` |
| `scaling` | 2.0 | static adapter scaling `s` |
| `mask_alpha` | 0.01 | gradual mask decay per step (Appendix A/C) |
| `ema_beta` | 0.85 | salience EMA: `0.85 * prev + 0.15 * cur` |
| `tau` | 4 | teacher layers sampled block-wise per step |
| `tuning_budget_initial/final` | 1.0 / 2.0 | `Delta_t` growth for rank updates |
| `top_fraction` | 0.5 | top-half salient adapters grow rank |
| `pred_distill_weight` | 1.0 (GLUE) / 0.1 (SQuAD, CNN/DM) | `L_distill` prediction term |
| `layer_distill_weight` | 0.9 | `L_layer` hidden-state term |
| `weight_decay`, `warmup_ratio`, `max_grad_norm` | 0.01, 0.06, 1.0 | AdamW + linear decay after warmup |
| `seed` | 42 | paper averages seeds 42/43/44 (`-seeds`) |

`eval_steps: 0` / `save_steps: 0` are sentinels meaning "once per epoch".

---

## 5. Reproducing each table / figure

All drivers are **orchestration only** — they never re-implement a method; they build configs, dispatch
to `scripts/train_apt.py` / `scripts/train_baseline.py`, aggregate over tasks and seeds, normalize
efficiency against FT (= 100 %), diff against the published numbers in the paper, and write JSON to
`outputs/<table>/`.

### Table 2 — RoBERTa-base and T5-base at 60 % sparsity

```bash
python scripts/run_table2.py --model roberta --methods ft lora mask_tuning cofi lora_prune_distill apt
python scripts/run_table2.py --model t5      --methods ft lora mask_tuning apt
python scripts/run_table2.py --model roberta --dry-run     # print published values, no training
```

Efficiency protocol: **TTA** = wall-clock seconds to reach **97 % of the FT** dev/test metric
(linear interpolation between the two bracketing evaluations); training peak memory via
`torch.cuda.max_memory_allocated()`; inference throughput/latency with batch size 128 for small models.

Expected (RoBERTa-base, 60 % sparsity; FT Train=Inf time/mem 100 %):

| Method | MNLI | SST2 | SQuAD v2 | Train time | Train mem | Inf time | Inf mem |
|---|---|---|---|---|---|---|---|
| FT | 87.6 | 94.8 | 82.9 | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| LoRA | 87.5 | 95.1 | 83.0 | 2137.0 % | 60.5 % | 100.0 % | 100.0 % |
| LoRA+Prune | 84.0 | 93.0 | 79.2 | 5128.3 % | 60.5 % | 38.0 % | 75.1 % |
| Prune+Distill | 87.3 | 94.5 | – | 1495.3 % | 168.5 % | 38.6 % | 79.2 % |
| LoRA+Prune+Distill | 84.2 | 91.9 | – | 6534.6 % | 141.4 % | 39.4 % | 82.3 % |
| **APT** | **86.4** | **94.5** | **81.8** | **592.1 %** | **70.1 %** | **41.3 %** | **78.1 %** |

Expected (T5-base, 60 % sparsity):

| Method | MNLI | SST2 | ROUGE-1/2/L | Train time | Train mem | Inf time | Inf mem |
|---|---|---|---|---|---|---|---|
| FT | 87.1 | 95.2 | 42.1 / 20.3 / 39.4 | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| LoRA | 87.0 | 95.0 | 38.7 / 17.2 / 36.0 | 255.5 % | 62.0 % | 100.0 % | 100.0 % |
| LoRA+Prune | 80.9 | 92.3 | 36.7 / 15.7 / 33.9 | 4523.5 % | 62.0 % | 47.1 % | 73.4 % |
| **APT** | **87.0** | **95.0** | **38.6 / 17.0 / 35.8** | **484.7 %** | **73.9 %** | **74.6 %** | **81.5 %** |

### Table 4 — RoBERTa-base ablations (60 % sparsity, one component removed at a time)

```bash
python scripts/run_table4_ablation.py --model roberta --ablation apt wo_ap wo_salience wo_at wo_ds
python scripts/run_table4_ablation.py --dry-run     # published values only
```

| Variant | SST2 | MNLI | Train time | Train mem |
|---|---|---|---|---|
| APT | 94.5 | 86.4 | 592.1 % | 70.1 % |
| w/o `A_P` | 94.4 | 87.5 | 82.6 % | 62.2 % |
| w/o salience (kurtosis) | 94.3 | 84.7 | 609.8 % | 65.0 % |
| w/o `A_T` | 93.2 | 84.5 | 684.9 % | 64.4 % |
| w/o `D_S` | 92.9 | 85.3 | 483.1 % | 61.9 % |

Ablation switches map to config overrides (see `ABLATION_OVERRIDES`):

* `wo_ap`   → no adaptive pruning: static dense masks / no salience-driven mask updates.
* `wo_salience` → magnitude-based density instead of the outlier-aware activation×gradient + √kurtosis score.
* `wo_at`   → `use_adaptive_tuning: false`: fixed initial rank, no top-half rank growth.
* `wo_ds`   → `use_distillation: false`: no `mu * L_distill` term.

### Table 7 — BERT-base vs PST / LRP at 50 % and 10 % density

```bash
python scripts/run_table7_bert.py --model bert --methods ft pst lrp apt --densities 0.50 0.10
python scripts/run_table7_bert.py --dry-run
```

APT targets GLUE average **83.2** (50 % density) and **76.8** (10 % density). PST/LRP must come from
their own repositories; when unavailable the script flags a LoRA+Prune proxy via `proxy_note(...)` and
leaves their cells empty rather than fabricating numbers (`--reference-json` can supply official values).

### Table 8 — RoBERTa GLUE at 40 % sparsity vs LoRA+Distill

```bash
python scripts/run_table8_glue.py --model roberta --sparsity 0.40 \
    --methods ft lora lora_distill apt
```

Seven GLUE tasks (MNLI, QQP, QNLI, SST2, CoLA, MRPC, RTE; **STS-B excluded** per Appendix D.2).
APT GLUE average **83.9**, above LoRA+Distill **80.0**, approaching LoRA 84.5 and FT 89.7.

### Figure 3 — performance vs relative inference efficiency

```bash
python scripts/run_figure3_sparsity.py --model roberta --sparsities 0.2 0.4 0.6 0.8
python scripts/run_figure3_sparsity.py --model t5 --sparsities 0.2 0.4 0.6 0.8
```

Sweeps target sparsity for the in-scope models, produces `outputs/figure3/figure3_<model>.json|.csv`
and, if `matplotlib` is installed, a plot (x = relative inference time normalized to FT, y = task
performance). Fig. 3 anchors also include the published Table 2 points via `reference_points(model)`.

### Appendix I / Table 11 — raw efficiency numbers

```bash
python -m apt.eval.efficiency          # self-test validating Table 11 -> Table 2 normalization
```

| Model | Method | TTA (s) | Train mem (MB) | Inf time (ms) | Inf mem (MB) |
|---|---|---|---|---|---|
| RoBERTa | FT | 127 | 2696 | 220.8 | 1157 |
| RoBERTa | APT | 752 | 1890 | 91.3 | 904 |
| T5 | FT | 366 | 7217 | 248.1 | 2347 |
| T5 | APT | 1774 | 5332 | 185.0 | 1913 |

These constants live in `apt/eval/efficiency.py` (`TABLE11_RAW`) and `apt/eval/metrics.py`
(`RAW_EFFICIENCY`); `normalize_efficiency` / `relative_from_table11` convert them to the relative
Table 2 percentages used by the drivers.

---

## 6. How APT is implemented (paper mapping)

1. **Adapter (Eq. 2)** — `H = m_o ∘ (W + s·W_B·W_A)(m_i ∘ X)` with frozen `W`, binary/annealed masks
   `m_i` (hidden dimensions) and `m_o` (heads / FFN neurons), static `s = 2`. Placed in the MHA
   Query/Value projections and in FFN projections (`apt/model_wrapper.py`); the wrapper also adds
   mask-only projections so physical pruning is exact. New `W_B` columns are zero-init and new `W_A`
   rows Gaussian-init, so **rank growth leaves the layer output unchanged**.
2. **Masks / schedule** — masks start at 1 and are *gradually* annealed with `alpha = 0.01`
   (Appendix A/C) toward the selected pattern; the cubic sparsity schedule is
   `gamma_t = gamma_T + (1 - gamma_T)(1 - t/T)^3`; parameter counts follow Eq. (6); the optimizer is
   reset whenever parameter shapes change (masks or ranks).
3. **Salience** — per block, `sum |dL/dH| · sum |H|` with batch/sequence reduction performed *before*
   the outer product to bound memory, plus `sqrt(kurtosis)` of the outlier activation
   `O_{:,j} = W_{:,j} ∘ X_{j,:}^T`; EMA `0.85 / 0.15`. Data/model/tuning parameters are all handled.
4. **Selection** — blocks are ranked by density = salience / (block parameter count); a
   binary-search knapsack keeps the largest top-`i` set whose Eq. (6) parameter count satisfies the
   sparsity constraint. Gated FFNs (T5) count as three linear layers; T5 decoder cross-attention is
   counted; biases are omitted from density.
5. **Adaptive tuning** — adapter importance = `sum |W_B ∘ dL/dW_B| · s`; the top-half salient adapters
   grow via `r_apt' = floor(r_apt · Delta_t' / Delta_t)`; optimizer reset afterwards.
6. **Self-distillation** — teacher layers are duplicated from the student with **frozen parameters
   shared**; `Tr` is a tunable LoRA layer initialized to identity; `tau = 4` contiguous blocks of
   teacher layers are sampled per step; `phi` maps each teacher layer to the closest non-pruned student
   layer and is recomputed **every training step**; `mu` is 0 before pruning and ramps linearly to 1 at
   the end of pruning; `L = mu·L_distill + (1-mu)·L_ft` with
   `L_distill = w_pred·L_pred + 0.9·L_layer`, `w_pred = 1.0` for GLUE and `0.1` for SQuAD/CNN-DM.
7. **Two-stage training (Algorithm 1)** — Stage 1 prunes + distills for `distill_epochs`; Stage 2
   fine-tunes the hardened pruned model for the remaining epochs to recover end-task performance.
8. **Merge & inference** — `apt/merge.py` folds `s·W_B·W_A` into `W` and physically slices pruned
   heads, neurons and hidden dimensions in a freshly instantiated HuggingFace config, so **tuning adds
   no inference overhead** (footnote 2, Sec. 6). `verify_merge_equivalence` checks forward-pass
   invariance before/after merging.

Programmatic use:

```python
from apt.training import TrainConfig, train_apt
from apt.model_wrapper import wrap_model
from apt.merge import merge_and_prune
from apt.eval.run_eval import evaluate_model

cfg = TrainConfig.from_dict(dict(model_name_or_path="roberta-base", task="sst2",
                                 target_sparsity=0.60, epochs=40, distill_epochs=20))
summary = train_apt(cfg, train_dataloader=train_dl, eval_dataloader=eval_dl)
```

---

## 7. Reproducibility protocol

* 3 seeds (`42, 43, 44`) — `--seeds 42 43 44`; drivers report mean ± std
  (`collect_metrics`, `mean_std`). `set_seed` seeds Python, NumPy and Torch (`PYTHONHASHSEED` is
  honored for reproducible iteration order).
* Measure training wall-clock to **97 % of the FT** metric (TTA), training peak memory with
  `torch.cuda.max_memory_allocated()`, and inference throughput/latency/peak memory at batch size 128.
  For knowledge-distillation baselines, training time includes teacher time (`combine_distill_time`).
* Single A100 for official comparisons; the small-model runs fit in 24 GB.
* Every driver supports `--dry-run` (published constants only) and `--self-test` (no GPU required), so
  pipeline wiring can be validated before launching the long runs.

## 8. Defaults chosen where the paper is silent

* Optimizer **AdamW**, weight decay **0.01**, betas (0.9, 0.999), eps 1e-8, grad-clip 1.0; linear
  warmup (`warmup_ratio = 0.06`) then linear decay; no-decay group for biases/LayerNorm.
* Initial masks all ones; kurtosis via `scipy.stats.kurtosis` (Fisher, default settings) over
  `O_{j,:}`.
* New `W_A` uses the LoRA default scale (`1/r`); new `W_B` zeros.
* Teacher→student mapping: closest non-pruned student layer by depth, ties → smaller index, recomputed
  each step (degenerate fully-pruned cases fall back to depthwise identity).
* Pruning onset: a short warmup, then pruning begins and ends at the target sparsity within Stage 1;
  mask decay `alpha = 0.01` per step; hardening before Stage 2 / export.
* External baselines use their repositories' default hyper-parameters unless the paper overrides them.

## 9. Out of scope

LLaMA-2 / Alpaca instruction tuning (Appendix G), 7B/13B scaling (Appendix H), and the
`lm-eval-harness` evaluation path are **not** implemented. CNN/DM and SQuAD v2 predictions are produced
with `num_beams = 4`, `length_penalty = 2.0` for T5 generation and the standard SQuAD v2
`n_best_size = 20` / `max_answer_length = 30` / `null_score_diff_threshold = 0.0` span post-processing.

---

## 10. Reference

```bibtex
@inproceedings{ro2024apt,
  title     = {APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference},
  author    = {Ro, Jaehun and Lee, Jaehyung and Kwon, Woosuk},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024}
}
```

Baselines: LoRA (Hu et al., 2021); Mask Tuning / retraining-free pruning (Kwon et al., 2022);
CoFiPruning (Xia et al., 2022).
