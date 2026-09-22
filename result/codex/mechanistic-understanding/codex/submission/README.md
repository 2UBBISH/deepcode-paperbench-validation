# A Mechanistic Understanding of Alignment Algorithms: DPO and Toxicity

Reproduction of *"A Mechanistic Understanding of Alignment Algorithms: A Case
Study on DPO and Toxicity"* (Lee, Bai, Pres, Wattenberg, Kummerfeld, Mihalcea;
ICML 2024), plus the two clarifications in the addendum (the probe is a matrix
``W_toxic in R^{d x 2}`` whose toxic column is ``W_toxic[:, 1]``, SVD is applied
to the *transpose* of the stacked value vectors, and DPO uses a 90:10 split) and
the scope notes (Llama2 is out of scope because its weights are gated; the
24,576-pair preference dataset *is* in scope).

Everything is implemented for **GPT2-medium**, which the paper uses for all of
its main-text results; the Llama2 code paths (GLU key/value vectors, gate
overrides, ``W_2`` scaling) are implemented but are not run by default.

---

## 1. What the repository contains

```
src/dpo_toxic/
  data/            Jigsaw, Wikitext-2 and RealToxicityPrompts loaders
  probe.py         Section 3.1  -- the linear toxicity probe W_toxic
  toxic_vectors.py Section 3.1  -- toxic MLP value/key vectors + SVD.U_toxic
  vocab_projection.py Section 3.2 -- projecting vectors onto the vocabulary
  interventions.py Section 3.3  -- x^{L-1} := x^{L-1} - alpha * W (+ alpha calibration)
  pplm.py          Section 4.2  -- PPLM guided by W_toxic
  pairs.py         Section 4.2  -- the 24,576 preference pairs
  dpo.py           Section 4.1  -- the DPO loss and trainer (Table 8 hyperparameters)
  analysis/        Section 5    -- parameter shift, activation drop, delta_x, PCA, logit lens
  unalign.py       Section 6    -- scaling toxic key vectors / turning gates back on
  architecture.py  model-agnostic access to keys, values, activations, residual streams
  evaluation/      toxicity (toxic-roberta), perplexity (Wikitext-2), F1
scripts/           one CLI per experiment (see the table below)
run_all.sh         the full paper-scale pipeline, in paper order
configs/paper_settings.json   every hyperparameter used
tests/smoke_test.py           end-to-end test on a small model
```

Each script is a thin CLI around the library:

| Paper section | Script | What it produces |
|---|---|---|
| 3.1 | `scripts/train_probe.py` (+ `scripts/extract_features.py`) | `W_toxic`, validation accuracy |
| 3.1 | `scripts/extract_toxic_vectors.py` | top-128 toxic value/key vectors, `SVD.U_toxic` |
| 3.2 | `scripts/project_vocab.py` | Table 1 / Table 6 analogue |
| 3.3 | `scripts/run_interventions.py` | Table 2 / Table 7 analogue + Table 3 prompts |
| 4.2 | `scripts/build_pairs.py` | the 24,576-pair dataset (PPLM negatives, greedy positives) |
| 4.1 | `scripts/train_dpo.py` | GPT2_DPO |
| 3.3/4 | `scripts/evaluate_model.py` | toxicity / PPL / F1 for any model |
| 5.1/5.2 | `scripts/analyze_dpo.py` | parameter shifts, activation drop, ``delta_x``, cosine histograms, PCA |
| Fig. 1 | `scripts/logit_lens.py` | layer-wise probability of ``sh*t`` |
| 6 | `scripts/unalign.py` | Table 4 (GPT2) / Table 5 (Llama2 code path) |
| Figs. | `scripts/make_figures.py` | figures 1-5 as PNGs |

## 2. Running it

```bash
pip install -e .                      # torch, transformers, datasets, scikit-learn, matplotlib
python tests/smoke_test.py --fast     # a few minutes on CPU

bash run_all.sh                        # full pipeline, paper settings (needs a GPU)
bash run_all.sh probe vectors vocab    # run individual stages
```

Everything writes to `artifacts/`; each stage reads the artifacts of the
previous stages, so stages can be re-run independently.

## 3. Method: how each paper section is implemented

### 3.1 Extracting toxic vectors

* **Probe.** `probe.extract_mean_residuals` runs GPT2-medium over Jigsaw and
  mean-pools the **last-layer residual stream over all timesteps**; a linear
  softmax probe (`ToxicityProbe`, weight shape `[2, d_model]`) is fit with
  Adam/cross-entropy on a 90:10 split. `W_toxic[:, 1]` -- the toxic row -- is
  saved as `artifacts/probe/w_toxic.pt`.
* **Toxic vectors.** For every layer, the rows of the ``[d_mlp, d_model]`` view
  of `W_V` are the value vectors and the rows of `W_K` the key vectors
  (`architecture.TransformerInternals`). All value vectors are ranked by cosine
  similarity with `W_toxic[:, 1]`; the top `N = 128` become
  `MLP.v_toxic` / `MLP.k_toxic`.
* **SVD.** The 128 unit value vectors are stacked into an `N x d` matrix and the
  SVD is computed on its transpose (addendum), giving `d`-dimensional
  `SVD.U_toxic[i]` (ordered by singular value).

### 3.2 Vocabulary projection

`vocab_projection.top_tokens` computes `E v` (the embedding matrix times a
vector) and reports the tokens with the largest dot product, for `W_toxic`, each
toxic value vector and each `SVD.U_toxic[i]` -- the paper's Table 1.

### 3.3 Interventions

The intervention is a forward pre-hook on the last layer's pre-MLP layernorm, so
`x^{L-1} := x^{L-1} - alpha * W` is applied to the residual stream itself and
propagates through the final MLP and the unembedding. `alpha` is calibrated by
bisection so that the Wikitext-2 perplexity matches the post-DPO model (23.34 for
GPT2), which is the paper's protocol for making the interventions comparable.
The three metrics are:

* **toxicity** -- maximum per-token probability of the "toxic" label from
  `unitary/unbiased-toxic-roberta` (the addendum's replacement for Perspective
  API) over 20 greedy tokens for the 1,199 RealToxicityPrompts *challenge*
  prompts (prompt toxicity >= 0.5, which is how the original release defines the
  challenge set);
* **perplexity** -- standard sliding-window Wikitext-2 perplexity;
* **F1** -- 2,000 Wikipedia sentences are split into a 10-token prompt and the
  true continuation; the model greedily generates as many tokens as the
  reference continuation and we report the harmonic mean of the bag-of-tokens
  precision and recall against that continuation (ConvAI2-style).

### 4.1 DPO

`dpo.dpo_loss` implements `-log sigma(beta log P - beta log N)` with the frozen
reference model; `dpo.train_dpo` uses the Table 8 hyperparameters (RMSprop,
lr 1e-6, batch 4, gradient accumulation 1, gradient clipping at norm 10, beta
0.1, validation patience 10) and a 90:10 split.

### 4.2 Preference pairs

For each Wikitext-2 sentence the positive continuation is greedy GPT2 sampling
and the negative continuation is generated by `pplm.PPLMGenerator`, whose
attribute classifier is exactly the toxicity probe (the probe takes the mean
residual stream of the last layer, and PPLM steers that same quantity). Step
size, top-k, iterations, geometric-mean and KL scales follow Table 9. Pairs whose
negative is not classified as toxic by the probe are discarded.

### 5. After DPO

* **5.1** `analysis.parameter_shift` reports the per-tensor cosine similarity and
  norm differences between GPT2 and GPT2_DPO.
* **5.2 / Figure 2** `analysis.activations` generates 20 tokens greedily with
  **GPT2** (addendum) for each challenge prompt and measures the mean activation
  `m_i = sigma(h . k_i)` of the toxic value vectors under both models.
* **5.2 / Figures 3-4** `analysis.residual_shift` collects `x^{l-mid}` (after
  attention, before the MLP), forms `delta_x = x_DPO - x_GPT2`, and projects the
  residuals onto `(mean delta_x, first principal component)`.
* **5.2 / Figure 5** the cosine similarity between the mean `delta_x^{19-mid}`
  and every shifted value vector `delta_MLP.v_i^j` (`j < 19`), together with the
  mean activation of those value vectors on RealToxicityPrompts.
* **Figure 1** `analysis.logit_lens` selects the prompts whose greedy next token
  is `sh*t` and applies the unembedding to every intermittent layer (both
  `l-mid` and post-block states).

### 6. Un-aligning

`unalign.scale_toxic_key_vectors` multiplies the key vectors of the 7 most toxic
value vectors by 10, enlarging `gamma(MLP.k_toxic)` so that the residual stream
re-enters the toxic regions. The Llama2 variants (`GateOverrideHook` sets
`sigma(W_1 x) = 1` for the top-8 gates, `scale_up_projection` multiplies `W_2`
by 3) are implemented but not run, since Llama2 is out of scope for this
reproduction.

## 4. Scope decisions and deviations

| Item | Paper | This reproduction | Why |
|---|---|---|---|
| Llama2-7b | full 5.2/6 analysis | code paths implemented, not run | gated weights -> out of scope per the addendum |
| Toxicity classifier | Perspective API | `unitary/unbiased-toxic-roberta` | addendum |
| Jigsaw | 561,808 comments | all rows of the HuggingFace mirror (159,571 + 306,328, deduplicated) | the mirror recommended in the addendum does not contain all 561,808 rows |
| Probe training | full 90:10 split | configurable; the included verification run used a subset (see section 6) | CPU-only environment, no GPU |
| Preference pairs | 24,576 pairs | implemented end-to-end; generation is GPU-bound and was not run here | CPU-only environment |
| DPO | GPT2_DPO, ~6,700 pairs to convergence | implemented; verified with a 4-step smoke run | CPU-only environment |
| F1 reference | "original Wikipedia continuation" | first 10 tokens as prompt, remaining tokens as reference | the paper does not specify the split point |
| `delta_x` in Fig. 5 | per-prompt shifts | mean shift over prompts (per-prompt shifts are also saved) | the paper's figure aggregates over prompts |

## 5. Verification performed in this environment

`tests/smoke_test.py` exercises every component on `distilgpt2` with tiny data:
internal accessors and hooks, probe training and serialisation, toxic-vector
extraction and SVD, vocabulary projection, DPO loss/training, PPLM-guided pair
construction, the three metrics, all Section 5 analyses, and the un-alignment
weight surgery. Results of the runs that were executed are recorded in
`artifacts/` and summarised in the next section.

Status at the time of writing: all 9 smoke tests pass (`python
tests/smoke_test.py`, ~25 minutes on CPU because of the generation steps;
`--fast` skips the two slowest).

## 6. Results obtained

The stages that are feasible on a CPU were run for real on GPT2-medium; the
full write-up is in [`docs/RESULTS.md`](docs/RESULTS.md).

| Paper result | This reproduction |
|---|---|
| Probe validation accuracy 94% (Section 3.1) | **93.8%** (trained on 3,000 of the ~466k available comments) |
| Toxic value vectors promote toxic tokens (Table 1, Section 3.2) | reproduced: `MLP.v_882^12` -> f\*\*k, s\*\*t, p\*\*s, hilar, stupidity (paper: f\*ck, sh\*t, piss, hilar, stupidity); `SVD.U_toxic[0]` -> stupidity, bullshit, p\*\*s, crap, a\*\*\*\*\*e, smug |
| GPT2 toxicity 0.453 (Table 2, NO OP) | **0.428** (60 RealToxicityPrompts challenge prompts, 20 greedy tokens, toxic-roberta) |
| GPT2 perplexity 21.7 (Table 2, NO OP) | **19.26** (first 400 Wikitext-2 test lines) |
| GPT2 F1 0.193 (Table 2, NO OP) | **0.138** (200 Wikipedia continuations; split point not specified by the paper) |
| SUBTRACT MLP.v_toxic: 0.305 at PPL 23.30 (Table 2) | **0.327 at PPL 23.05** with `MLP.v_882^12` at alpha 60 |
| DPO, PPLM pairs, Sections 5 and 6 | implemented and smoke-tested; they require GPU-scale compute (see below) |

The complete set of commands used for the numbers above is listed in
`docs/RESULTS.md`; the generated artifacts live in `artifacts/` (gitignored) and
can be regenerated with `run_all.sh`.

## 7. Compute requirements for the paper-scale run

| Stage | Cost | Notes |
|---|---|---|
| Probe feature extraction | ~500k GPT2-medium forwards | ~2 GPU-hours at batch 64 |
| PPLM pairs (24,576) | 24,576 generations x 50 gradient steps | the dominant cost; parallelise across GPUs |
| DPO | ~6,700 pairs to convergence | minutes-to-hours on one GPU |
| Sections 5-6 analyses | ~1,199 prompts x a few forward passes per layer | minutes on GPU |

The repository therefore ships the full pipeline (``run_all.sh``) plus reduced
settings (`--max-train`, `--max-prompts`, `--shard-size`, `--n-pairs`,
`--pplm-num-iterations`, `--max-steps`) so that every stage can also be run
end-to-end on a laptop for verification.
