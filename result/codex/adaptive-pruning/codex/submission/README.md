# APT — Adaptive Pruning and Tuning of Pretrained Language Models

Reproduction of

> Bowen Zhao, Hannaneh Hajishirzi, Qingqing Cao.
> **APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient
> Training and Inference.** ICML 2024.

The repository implements the full APT method *and* the baselines it is compared
against, together with the data pipeline, the efficiency measurements and the
ablation/analysis studies reported in the paper.

---

## 1. What the paper does and where it lives in the code

APT adaptively prunes **and** tunes a pretrained LM during fine-tuning so that
both training and inference become cheaper without losing task accuracy.

| Paper element | Where it is implemented |
|---|---|
| APT adapter, Eq. (2): `H = m_o ∘ (W + s·W_B W_A) X ∘ m_i` | [`apt/adapter.py`](apt/adapter.py) (`APTLinear`) |
| Prunable blocks: MHA heads, FFN neurons, hidden dimensions | [`apt/wrap.py`](apt/wrap.py) (`Topology`, `wrap_roberta`, `wrap_t5`) |
| Outlier-aware salience, Eq. (3)–(5) + Appendix B | `APTLinear.compressed_salience`, `APTLinear.tuning_salience*`, `compute_block_salience`, `kurtosis_from_moments` |
| Efficient mask search (binary search over salience density), Eq. (6) + Appendix C | [`apt/blocks.py`](apt/blocks.py) (`PruningState.select_for_budget`) |
| Gradual mask annealing with `alpha = 0.01` | `PruningState.anneal_masks` |
| Adaptive tuning: salience-based rank growth, Section 4.3 | [`apt/tuning.py`](apt/tuning.py) |
| Efficient self-knowledge distillation, Eq. (7), Section 4.4 | [`apt/distill.py`](apt/distill.py) |
| Cubic sparsity schedule + `mu` ramp, Appendix A | [`apt/schedule.py`](apt/schedule.py) |
| Two-stage training loop (prune+distill, then recover on the *pruned* model) | [`apt/trainer.py`](apt/trainer.py) |
| Physical pruning → real inference speedups | [`apt/physical.py`](apt/physical.py) |
| Efficiency metrics (TTA, throughput, peak memory) | [`apt/trainer.py`](apt/trainer.py) (`TimeToAccuracy`), [`apt/measure.py`](apt/measure.py) |
| Tasks: GLUE (SST2/MNLI/…), SQuAD v2.0, CNN/DailyMail | [`data/tasks.py`](data/tasks.py) |
| Baselines: FT, LoRA, LoRA+Prune, Prune+Distill, LoRA+Prune+Distill | [`baselines/`](baselines) |
| Experiment entry points | [`scripts/run.py`](scripts/run.py) + [`scripts/run_table2.sh`](scripts/run_table2.sh), [`scripts/run_table4.sh`](scripts/run_table4.sh), [`scripts/run_figure3.sh`](scripts/run_figure3.sh) |

### Scope

Following the task addendum, the following are **out of scope** and are *not*
implemented: everything LLaMA-related (Alpaca instruction tuning and the
`lm-eval-harness` Open-LLM-leaderboard evaluation), the Appendix G distillation
strategy comparison and the Appendix H adaptive-pruning/tuning analysis. All
RoBERTa / T5 experiments that appear in the main text, including the GLUE,
SQuAD and CNN/DM result tables and the Table 4 ablations, are in scope and are
implemented.

---

## 2. Method implementation notes

### 2.1 APT adapter (`apt/adapter.py`)

`APTLinear` wraps a frozen `nn.Linear` and adds

* an input mask `m_i` (indexed by the input feature) and an output mask `m_o`
  (indexed by the output feature) — both registered as buffers and applied as
  Hadamard products, and
* a LoRA branch `s · W_B W_A` with a *dynamic* rank `r_apt`.

`grow_rank` increases `r_apt` while reproducing the LoRA initialisation for the
new entries (`W_A ~ N(0, σ²)`, `W_B` zero padded) so the layer output is
unchanged, exactly as required by Section 4.3.  `merge_lora` folds the trained
branch back into `W` for inference.

The salience statistics are collected with a forward pass (sums of `|activation|`
and of `|gradient|` along the batch and sequence axes — Algorithm 1) plus a
`register_full_backward_hook` for `dL/dX`, and with the parameters' `.grad` for
the tuning-weight term of Appendix B.  Accumulation is done in `float64` on the
CPU.

### 2.2 Block table and the parameter-count formula

Every feature of every wrapped linear belongs to exactly one structural block
(`PruningState.check_coverage` enforces this).  Head/neuron blocks partition the
**rows** of a weight while dimension blocks partition its **columns**, so the
size of the retained sub-network is *not* the sum of the retained block sizes —
the paper's Eq. (6),

```
C(top-i) = Σ_g d_g' · (4 · d_h · n_h^g + k_ff · n_f^g)
```

is used instead (`PruningState._prefix_costs`).  It reproduces the true model
size exactly at full density — verified in `tests/test_wrap.py`:

* RoBERTa-base: 84,934,656 transformer parameters,
* T5-base: 198,180,864 (encoder + decoder) parameters,

matching the values implied by Appendix C.  `k_ff = 3` for the gated FFN of
`t5-lm-adapt`, `k_ff = 2` for RoBERTa/T5-base.  The prefix costs are computed in
vectorised cumulative sums, so the binary search of Appendix C is exact and
cheap (~42 ms for the 37,776 blocks of RoBERTa-base, which is why Algorithm 1's
per-step re-selection is the default: `adjust_interval = 1`).

### 2.3 Salience

`compute_block_salience` implements Eq. (5): the compressed activation×gradient
product of a block plus `sqrt(Kurt(·))` of the corresponding activation
distribution.  Kurtosis is obtained from streaming raw moments
(`kurtosis_from_moments`, validated against `scipy.stats.kurtosis` in the unit
tests).  Turning off the kurtosis term reproduces the `w/o kurtosis` ablation of
Table 5.  The score is computed with a precomputed `feature → block` scatter
plan (`_salience_plan`) and `index_add_`, which keeps a RoBERTa-base step cheap
instead of walking 37,776 blocks in Python.

### 2.4 Adaptive tuning

Adapters are ranked by `I(H_apt) = Σ |W_B · dL/dW_B|` and the ranks of the
top-half salient adapters are linearly increased from `r = 8` up to
`target_rank` (default 64) during the pruning stage, after which the optimizer
is reset — the paper resets it "every time after each parameter size changes".
`--target-rank 8` therefore disables adaptive tuning without touching the rest
of the algorithm, and the `wo_salience` ablation grows *all* adapters instead of
the salient ones (the "equally increasing parameters across all layers" ablation
of Section 5.6).  The ablation switches in `apt/pipeline.py::ABLATIONS` map
directly onto the Tables 4 and 5 rows:

| Ablation key | Configuration | Paper row |
|---|---|---|
| `apt` | defaults | APT |
| `wo_adaptive_pruning` | no masks, no distillation | w/o `A_P` |
| `wo_adaptive_tuning` | static ranks | w/o `A_T` |
| `wo_distillation` | no `L_distill` | w/o `D_S` |
| `wo_salience` | grow every adapter, not the salient ones | w/o salience |
| `wo_kurtosis` | drop the `sqrt(Kurt)` term | w/o kurtosis (Table 5) |

### 2.5 Self-distillation

The teacher is the **same network evaluated without the pruning masks**:
frozen parameters are shared, only the tuning layers are duplicated
(`TeacherCache`) and `Tr` is an identity-initialised low-rank transform
(`IdentityLoRA`).  Because the teacher is only a different mask setting, no
second copy of the LM is needed — this is what removes the "teacher next to the
student on the GPU" cost of CoFi.  The set `T` of teacher layers is sampled
block-wise at random every step (RAIL-KD) and the mapping `phi` is recomputed
each step, as the addendum requires.  Weights (Eq. 7):

* GLUE: `L_distill = L_pred + 0.9 · L_layer`
* SQuAD / CNN-DM: `L_distill = 0.1 · L_pred + 0.9 · L_layer`
* overall: `L = mu · L_distill + (1 - mu) · L_ft` with
  `mu = min(1, (t - start)/(end - start))`.

The teacher forward is executed *before* the student forward so that no buffer
is mutated between the student's forward and its backward pass; the mask swap
itself is done by rebinding buffers rather than in-place writes.

### 2.6 Sparsity schedule

Appendix A writes `gamma_t = gamma_T + (1 - gamma_T)(1 - t/T)^3`.  Read
literally this decreases from 1 to `gamma_T`, i.e. `gamma` denotes the
*retained* ratio; equivalently the sparsity ramps as
`gamma_T^sparsity · (1 - (1 - t/T)^3)` (identical expression).  The ramp is
fastest at the beginning, which matches the paper's "early pruning" description;
`tests/test_schedule.py` checks this behaviour and the exact formula.
Because the ramp only reaches `gamma_T` at `t = T` while the loop stops at
`T - 1`, the blocks are re-selected once more for the *target* sparsity before
the masks are hardened, so the returned model always satisfies
`1 - C(Theta_T, M_T)/C(Theta_0, M_0) >= gamma_T`.

### 2.7 Physical pruning (`apt/physical.py`)

Inference speedups require baking the masks in.  `materialize_*` merges the
trained adapter into `W`, slices the weights down to the retained heads /
neurons / dimensions and rebuilds a smaller model:

* RoBERTa/BERT: the head count is set *per layer*
  (`num_attention_heads`, `attention_head_size`, `all_head_size`), task heads,
  pooler, embeddings and LayerNorms are re-indexed.
* T5: the encoder-decoder shares one hidden dimension.  Because the relative
  position bias is computed once per stack and reused, a stack must contain a
  uniform number of heads; a head is kept in a stack when it survives in at
  least half of that stack's attention modules.  Encoder self-attention, decoder
  self-attention and decoder cross-attention are three independent stacks.

Smoke test on `roberta-base` at 60 % sparsity: 125 M → 50.1 M parameters.

### 2.8 Baselines (`baselines/`)

* **FT** — full fine-tuning, the reference for every normalised metric.
* **LoRA** — standard LoRA (`LoRALinear`), merged before inference.
* **LoRA+Prune** — Mask Tuning (Kwon et al.) adapted to a LoRA-tuned model, as
  the addendum requests: the LoRA-tuned model is merged, per-unit Fisher
  information `E[(dL/dW)²]` is estimated on a calibration set, units are removed
  least-importance-per-parameter first until the budget is met (mask search),
  continuous mask variables are tuned with the weights frozen (mask tuning), the
  model is physically pruned and finally retrained.
* **Prune+Distill (CoFi)** — hard-concrete L0 gates on heads/neurons/dimensions
  plus layer-wise hidden-state distillation from a *separate full-size teacher*
  (`baselines/cofi.py`), optionally with only the L0 modules and LoRA parameters
  trainable (**LoRA+Prune+Distill**, the addendum's requirement).

---

## 3. Running the experiments

```bash
pip install -r requirements.txt
export PYTHONPATH=.

# APT on RoBERTa-base / SST2 at 60% sparsity (Table 2)
python scripts/run.py apt --task sst2 --model roberta-base --sparsity 0.6 \
    --epochs 40 --distill-epochs 20 --output-dir runs/rb_sst2_apt

# the whole of Table 2 (RoBERTa + T5, all baselines)
bash scripts/run_table2.sh

# Table 4 ablations and Table 8 GLUE comparison
bash scripts/run_table4.sh

# Figure 3 sparsity analysis
bash scripts/run_figure3.sh

# aggregate every runs/*/result.json into one table
python scripts/collect_results.py --glob 'runs/*/result.json'
```

`scripts/run.py --dry-run` prints the resolved configuration without training.
`--limit-train/--limit-eval` subsample the data for quick sanity checks.

Training hyper-parameters default to Table 6 of the paper (LR `2e-4`, batch 32,
40 epochs of which 20 are distillation epochs for GLUE; `1e-4`, batch 16, 16/6
epochs for CNN/DM).

### Environment notes

* The reference configuration is a single A100, `fp32`/`fp16`.  The code runs on
  CUDA, MPS and CPU; peak memory uses `torch.cuda.max_memory_allocated()` when a
  GPU is available and the process high-water mark otherwise.
* No long experiment was executed here.  What *was* executed:
  * `pytest tests/` — 37 unit tests (adapter semantics, block partitioning,
    parameter counting, schedules, distillation weights, task/span
    construction, physical pruning, baselines), all passing;
  * `tests/smoke_test.py` and `tests/smoke_test_t5.py` — full
    prune+distill → physical pruning → recover → efficiency-measurement runs on
    a 2-layer toy RoBERTa / T5;
  * a short run of the real `roberta-base` on SST2 with 48 training examples
    which completes the same pipeline end to end (125 M → 50.1 M parameters).

---

## 4. Results obtained during development (smoke scale only)

These numbers come from deliberately tiny runs (they start from the *pretrained*
model, so accuracies are far from converged) and only demonstrate that the
pipeline behaves as intended:

| Run | Setting | Result |
|---|---|---|
| toy RoBERTa (`tests/smoke_test.py`) | 2 epochs, target sparsity 0.6 | achieved sparsity 0.602, 3.34 M → 2.63 M parameters |
| toy T5 (`tests/smoke_test_t5.py`) | 2 epochs, target sparsity 0.6 | achieved sparsity 0.604, 2.26 M → 1.42 M parameters, generation-based accuracy 0.75 |
| `roberta-base`, SST2, 48 train examples | 2 epochs (1 distill), target sparsity 0.6 | achieved sparsity 0.599, 125 M → 50.1 M parameters, inference measurement produced |
| baselines on the toy RoBERTa | 60 % target, 2 epochs | FT 0.688 accuracy; LoRA runs; LoRA+Prune 0.629 sparsity; CoFi and LoRA+Prune+Distill 0.602 sparsity |

The full runs are launched with the `scripts/run_*.sh` entry points described
above; their results should be compared to Tables 2–5 of the paper.

For reference, the numbers a successful reproduction should land near:

| Paper table | Setting | Reference values |
|---|---|---|
| Table 2 | RoBERTa-base, 60 % sparsity, SST2 | FT 94.8, LoRA 95.1, LoRA+Prune 93.0, Prune+Distill 94.5, LoRA+Prune+Distill 91.9, **APT 94.5** |
| Table 2 | RoBERTa-base, 60 % sparsity, MNLI | FT 87.6, LoRA 87.5, LoRA+Prune 84.0, Prune+Distill 87.3, LoRA+Prune+Distill 84.2, **APT 86.4** |
| Table 2 | RoBERTa-base, 60 % sparsity, SQuAD v2 (F1) | FT 82.9, LoRA 83.0, LoRA+Prune 79.2, **APT 81.8** |
| Table 2 | T5-base, 60 % sparsity, SST2 / MNLI / CNN-DM | FT 95.2 / 87.1 / 42.1-20.3-39.4, LoRA+Prune 92.3 / 80.9 / 36.7-15.7-33.9, **APT 95.0 / 87.0 / 38.6-17.0-35.8** |
| Table 2 | Training cost (relative to FT) | APT 592 % *time-to-accuracy* and 70.1 % peak training memory vs 5128 % / 60.5 % for LoRA+Prune |
| Table 2 | Inference cost (relative to FT) | APT 41.3 % time and 78.1 % memory at 60 % sparsity |
| Table 4 | RoBERTa-base ablations | APT 94.5 / 86.4; w/o `A_P` 94.4 / 87.5; w/o `A_T` 93.2 / 84.5; w/o `D_S` 92.9 / 85.3; w/o salience 94.3 / 84.7 |
| Table 5 | LLaMA 2 7B ablations | out of scope for this reproduction |
| Figure 3 | sparsity sweep | APT should dominate the LoRA+Prune baseline in the accuracy/efficiency plane at every sparsity |

The normalisation used by Tables 2/11 is "smaller is better" for time and
memory: a value of `x %` means the method costs `x %` of fine-tuning.

### Caveats of this reproduction

* The results above are *not* reproduced here (no GPU is available in this
  environment); the repository is written so that they can be produced by the
  scripts, and every design decision is documented in the sections above.
* T5 inference efficiency is expected to be *worse* than the LoRA+Prune baseline
  (Section 5.4 explains why: APT prunes more decoder parameters, which are
  computationally cheaper for classification), so a reproduction that makes T5
  *faster* than the baseline would be suspicious.
* The T5 physical-pruning path enforces a uniform head count inside each
  attention stack because the relative position bias is shared across the whole
  stack (see §2.7).  Pretraining-time behaviour is unaffected; only the shapes of
  the materialised inference model change.

---

## 5. Repository layout

```
apt/            APT method (adapter, blocks/salience/search, tuning, distillation,
                schedules, trainer, physical pruning, efficiency measurement)
baselines/      FT, LoRA, Mask Tuning (LoRA+Prune), CoFi (Prune+Distill, LoRA+Prune+Distill)
data/           dataset/tokenisation/task adapters for GLUE, SQuAD v2.0 and CNN/DM
scripts/        CLI + shell wrappers reproducing Tables 2/4 and Figure 3
tests/          unit tests (no downloads required: configs are built locally)
```

## 6. Reference

```bibtex
@inproceedings{zhao2024apt,
  title     = {APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference},
  author    = {Zhao, Bowen and Hajishirzi, Hannaneh and Cao, Qingqing},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year      = {2024}
}
```
