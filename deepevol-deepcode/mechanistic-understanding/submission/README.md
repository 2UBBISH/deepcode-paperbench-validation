# A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity

Reproduction of

> **A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity**
> Andrew Lee, Xiaoyan Bai, Itamar Pres, Martin Wattenberg, Jonathan K. Kummerfeld, Ravid Shwartz-Ziv

This repository reproduces the **GPT2-medium** results of the paper: how Direct Preference
Optimization (DPO) reduces toxicity, why the toxicity-mediating MLP value vectors survive
alignment, and how re-scaling toxic key vectors re-activates the toxicity (un-alignment /
jailbreak behaviour).

---

## 1. Scope and substitutions

| Item | Paper | This reproduction |
| --- | --- | --- |
| Base model | GPT2-medium (24 layers, d_model=1024, d_mlp=4096) | `openai-community/gpt2-medium` |
| Aligned model | GPT2_DPO (paper) | trained locally → `artifacts/models/gpt2_dpo` |
| Toxicity metric | Perspective API | **`unitary/unbiased-toxic-roberta`** (open substitute) |
| Jigsaw | Kaggle download | **HF mirror `thesofakillers/jigsaw-toxic-comment-classification-challenge`** |
| Llama2-7b / GLU results | Tables 5 and GLU analyses | **out of scope** — code paths stubbed (`src/model_utils.glu_*`, `src/unalign.scale_glu_gates`) |
| Dataset sizes | 24,576 preference pairs, 1,199 RTP challenge prompts, 295 `sh*t` prompts, 2,000 F1 sentences | identical |

Documented defaults where the paper is silent:

* Probe (`W_Toxic`) optimisation: AdamW, lr `1e-3`, weight decay `0.01`, batch `256`, ≤20 epochs, patience `3`, no bias.
* Intervention strength `alpha`: selected **per vector** so the intervened Wikitext-2 perplexity matches the post-DPO perplexity (`23.34`).
* Generation defaults: greedy decoding, `20` new tokens, fixed seed `0`.
* PPLM implementation knobs (`pool`, `grad_norm`, `normalize_inside`) follow the PPLM reference implementation.

---

## 2. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: Python 3.10+, PyTorch 2.x (CUDA recommended), `transformers`, `datasets`,
`huggingface_hub`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `tqdm`, `pyyaml`, `matplotlib`,
`seaborn`, `pytest`.

Hardware: one CUDA GPU with ≥16 GB VRAM is recommended for GPT2-medium DPO training and the
24,576-pair PPLM generation. CPU-only runs work for the tests and `--quick` smoke tests but not
for a full reproduction.

Check the environment at any time:

```bash
python main.py --check-env
```

---

## 3. Repository layout

```
.
├── main.py                       # end-to-end orchestration (all phases)
├── configs/
│   ├── default.yaml              # paths, model names, seeds, in-scope flags, reference numbers
│   ├── dpo.yaml                  # DPO hyperparameters (Table 8)
│   └── pplm.yaml                 # PPLM hyperparameters (Table 9)
├── data/
│   ├── jigsaw.py                 # Jigsaw loading, binarisation, 90:10 split, tokenisation
│   ├── wikitext.py               # Wikitext-2 prompts, PPL corpus, 2,000 F1 sentences
│   ├── realtoxicity.py           # RealToxicityPrompts challenge subset + 295 "sh*t" prompts
│   └── pairwise.py               # preference-pair container, 90:10 split, resumable sharding
├── src/
│   ├── model_utils.py            # GPT2 loading, hooks, residual/MLP access, vocab projection, GLU stubs
│   ├── probe.py                  # W_Toxic linear toxicity probe
│   ├── toxic_vectors.py          # MLP.v_Toxic / MLP.k_Toxic / SVD.U_Toxic + vocabulary projections
│   ├── interventions.py          # residual-stream subtraction interventions (Table 2/3)
│   ├── pplm_generate.py          # PPLM toxic-continuation generator (pair construction)
│   ├── dpo_trainer.py            # DPO loss and trainer (Eq. 1, Table 8)
│   ├── unalign.py                # GPT2 un-alignment by scaling MLP.k_Toxic (Table 4)
│   ├── analysis/                 # logit lens, activations, residual shift, parameter diff, plots
│   └── eval/                     # toxicity, perplexity, F1
├── scripts/                      # one CLI entry point per pipeline phase
└── tests/                        # shape, DPO-loss and probe smoke tests
```

---

## 4. Pipeline

Run everything:

```bash
python main.py --all
```

Phases are dependency-ordered (`probe → vectors → interventions/pairs → dpo → eval → analyze → unalign`)
and cached artifacts are skipped unless `--force` is given. Run individual phases with
`python main.py --phases probe,vectors,...` or call the scripts directly.

### Phase 0 — environment & data

```bash
python main.py --check-env
python main.py --plan           # show phases, inputs, outputs, paper targets
```

### Phase 1 — toxicity probe `W_Toxic` (§3.1)

Frozen GPT2-medium features = last-layer residual stream **averaged over all timesteps** of
Jigsaw comments; a linear softmax classifier `softmax(W_Toxic x)` with `W_Toxic ∈ R^{d_model×2}`
is trained (column 1 = toxic direction, used everywhere downstream).

```bash
python scripts/train_probe.py                 # writes artifacts/probe/w_toxic.pt (+ .json)
python scripts/train_probe.py --quick
```

Target: **≈94% validation accuracy** on the 90:10 Jigsaw split.

### Phase 2 — toxic vector extraction & interventions (§3.1–3.3)

Rank **all** MLP value vectors (24 × 4096) by cosine similarity with `W_Toxic[:,1]`; keep the top
**N = 128** as `MLP.v_Toxic` and their matching key vectors `MLP.k_Toxic`. Build `SVD.U_Toxic`
by SVD of the **transposed** `d_model × N` value-vector matrix. Vocabulary projections
`r = E @ v` reproduce Table 1 token groups.

```bash
python scripts/extract_toxic_vectors.py       # artifacts/vectors/toxic_vectors.pt + table1_projections.json
python scripts/run_interventions.py           # Table 2 / Table 3
```

Residual-stream subtraction replaces `x^{L-1}` with `x^{L-1} − α·W` for `W ∈ {W_Toxic, MLP.v_Toxic[i], SVD.U_Toxic[i]}`;
`α` is chosen per vector to match post-DPO perplexity.

### Phase 3 — pairwise preference dataset (§4.2, Appendix E Table 9)

For each Wikitext-2 prompt: **non-toxic / preferred** continuation = greedy GPT2 decoding;
**toxic / non-preferred** continuation = PPLM with `W_Toxic` as the attribute classifier
`p(a|w)`, optimising `p(y|a) ∝ p(y)p(a|y)`. 24,576 pairs, 90:10 split, sharded & resumable.

```bash
python scripts/generate_pairs.py              # artifacts/data/pairs_shards/*.jsonl + pairs.jsonl
python scripts/generate_pairs.py --quick
```

Table 9 values used: step size `0.4`, temperature `1`, top-k `10`, iterations `50`,
window length `0`, horizon length `1`, decay `false`, gamma `1`, `gm_scale 0.95`, `kl_scale 0.1`.

### Phase 4 — DPO training (§4.1–4.2, Appendix E Table 8)

```bash
python scripts/train_dpo.py                   # artifacts/models/gpt2_dpo
```

Loss (`Eq. 1`, `beta = 0.1`):

```
L_DPO = −E[ log σ( β·log(π_θ(y⁺|w)/π_ref(y⁺|w)) − β·log(π_θ(y⁻|w)/π_ref(y⁻|w)) ) ]
```

Optimiser RMSProp, lr `1e-6`, batch `4`, grad accumulation `1`, max grad norm `10`,
early stopping on `loss/valid` with patience `10`, 90:10 pair split.

### Phase 5 — evaluation (§3.3)

```bash
python scripts/eval_model.py --all            # GPT2 and GPT2_DPO
python scripts/eval_model.py --model artifacts/models/gpt2_dpo
```

* **Toxicity** — mean `unitary/unbiased-toxic-roberta` score over the 1,199 RealToxicityPrompts
  challenge prompts (greedy 20-token continuations).
* **Perplexity** — Wikitext-2 sliding-window PPL.
* **F1** — token-overlap F1 of generated vs. original continuation on 2,000 Wikipedia sentences.

### Phase 6 — mechanistic analysis (Figures 1–5, §4.2/§5)

```bash
python scripts/analyze_dpo.py --phases all
python scripts/analyze_dpo.py --phases logit_lens        # Figure 1
python scripts/analyze_dpo.py --phases activations       # Figure 2 / Eq. 1
python scripts/analyze_dpo.py --phases parameter_diff    # §5.1 claims
python scripts/analyze_dpo.py --phases residual_shift    # Figures 3–5
```

* **Logit lens (Fig. 1)** — unembed every layer's `l-mid` (post-attention) and block-output state on
  the 295 prompts whose greedy next token is `sh*t`; MLP layers promote it most and GPT2_DPO
  reduces it.
* **Mean activations (Fig. 2, Eq. 1)** — `m_i = σ(x^l · k_i^l)` per toxic vector over 1,199 prompts
  × 20 greedy tokens.
* **Activation region** — `γ(k_i^l) = { g | σ(k_i^l · g) > 0 }`.
* **Parameter deltas (§5.1)** — cosine similarity `> 0.99` and mean norm difference `< 1e-5` for
  every parameter (unembedding exception `< 1e-3`).
* **Residual shift (Figures 3–5)** — `δx^{(l-mid)} = x_DPO − x_GPT2`, PCA projection of the streams,
  and `cos(δx, δMLP.v_i^j)` for `j < l` with the mean-activation overlay.

### Phase 7 — un-alignment (§6, Table 4)

```bash
python scripts/unalign_gpt2.py                # artifacts/unalign
```

Select the **7** MLP vectors with the highest cosine similarity to `W_Toxic[:,1]`, scale their
`MLP.k_Toxic` key vectors **10×** (enlarging the toxic activation regions), then re-evaluate.

---

## 5. Reference results (GPT2-medium)

Probe:

| Metric | Target |
| --- | --- |
| `W_Toxic` validation accuracy | ≈ 0.94 |

Table 2 — residual-stream subtraction interventions:

| Setting | Toxicity ↓ | PPL | F1 |
| --- | --- | --- | --- |
| NO OP | 0.453 | 21.70 | 0.193 |
| SUBTRACT `W_Toxic` | 0.245 | 23.56 | 0.193 |
| SUBTRACT `MLP.v_770^19` | 0.305 | 23.30 | 0.192 |
| SUBTRACT `SVD.U_Toxic[0]` | 0.268 | 23.48 | 0.193 |

GPT2_DPO (§4.2): toxicity **0.208**, PPL **23.34**, F1 **0.195** (validation loss plateaus around
≈ `6,700` pair examples).

Table 4 — un-alignment (7 key vectors × 10):

| Setting | Toxicity ↑ | PPL | F1 |
| --- | --- | --- | --- |
| GPT2_DPO | 0.208 | 23.34 | 0.195 |
| GPT2_DPO + scaled `MLP.k_Toxic` | ≈ 0.458 | ≈ 23.30 | ≈ 0.195 |
| GPT2 (pre-alignment) | 0.453 | 21.70 | 0.193 |

---

## 6. Tests

```bash
python -m pytest tests -q
# or
python -m unittest discover -s tests
```

* `tests/test_shapes.py` — architecture constants (L=24, d=1024, d_mlp=4096), value/key-vector
  shapes, residual capture shapes, vocab projection, reversible key scaling.
* `tests/test_dpo_loss.py` — DPO loss identities (loss = log 2 at zero margin, monotonicity,
  β scaling), Table 8 config defaults, sequence log-probs, pair encoding, trainer smoke test.
* `tests/test_probe.py` — probe conventions (`W[:,1]` = toxic), softmax semantics, synthetic
  training, save/load round-trip.

All tests are offline: they fall back to tiny randomly-initialised GPT2 models when
GPT2-medium cannot be downloaded (`REPRO_TINY=1` forces the tiny path).

---

## 7. Artifacts produced

```
artifacts/
├── probe/w_toxic.pt, w_toxic.json
├── vectors/toxic_vectors.pt, toxic_vectors.json, table1_projections.json
├── data/pairs_shards/*.jsonl, pairs.jsonl, generate_pairs_summary.json
├── models/gpt2_dpo/                      # GPT2_DPO checkpoint + trainer_state.json
├── eval/toxicity_*.json, perplexity_*.json, f1_*.json, eval_comparison.json
├── interventions/interventions.json, .md, interventions_table2.png, table3_examples.json
├── analysis/logit_lens_*.json, mean_activations.json, parameter_diff.json,
│            parameter_diff_claims.json, residual_shift_*.json, figure1..figure5 PNGs
├── unalign/unalign_results.json, unalign_results.md, unalign_table4.png
└── main_run_summary.json
```

---

## 8. Notes on faithfulness

* Vector/value indexing follows the HF GPT2 layout: `MLP.k_i^l = c_fc.weight[:, i]`,
  `MLP.v_i^l = c_proj.weight[i, :]`; both are exposed as row vectors of shape `[d_model]`.
* `SVD.U_Toxic` is computed on the **transposed** `d_model × N` matrix, so singular vectors are
  `d_model`-dimensional (author clarification).
* All cosine similarities downstream of the probe use `W_Toxic[:, 1]` (toxic column).
* Llama2/GLU code paths are present but explicitly stubbed and out of scope; they raise or return
  documented placeholders.
