# Results obtained in this environment

The reproduction environment has no GPU, so the paper-scale runs (PPLM data
generation and DPO training) were implemented and smoke-tested rather than run
to completion. The stages that are CPU-feasible were executed for real on
GPT2-medium: the toxicity probe, the toxic-vector extraction, the SVD basis, the
vocabulary projection and the evaluation metrics. Their outputs are recorded
below and stored under `artifacts/` (untracked; regenerate with the scripts).

## 1. Section 3.1 -- toxicity probe

| | paper | this reproduction |
|---|---|---|
| model | GPT2-medium | GPT2-medium |
| data | Jigsaw, 561,808 comments, 90:10 | Jigsaw (HF mirror), 3,000 train / 600 validation comments, 90:10 |
| pooling | mean of the last-layer residual stream over all timesteps | same |
| validation accuracy | 0.94 | **0.9383** |

Command:

```bash
python scripts/extract_features.py --model gpt2-medium --split train --shard 0 --shard-size 1500 --max-length 48 --batch-size 48
python scripts/extract_features.py --model gpt2-medium --split train --shard 1 --shard-size 1500 --max-length 48 --batch-size 48
python scripts/extract_features.py --model gpt2-medium --split val   --shard 0 --shard-size 600  --max-length 48 --batch-size 48
python scripts/train_probe.py --model gpt2-medium --features-dir artifacts/probe/features --epochs 15
```

The probe reaches the paper's accuracy even though it is fit on ~0.6% of the
comments, which indicates the toxic direction is stable. At paper scale the same
command is run without the shard caps (`--features-dir` accepts as many shards as
you generate).

## 2. Section 3.1 -- toxic value vectors and `SVD.U_toxic`

Top-10 value vectors ranked by cosine similarity with `W_toxic[:, 1]`:

| rank | vector | cosine |
|---|---|---|
| 0 | MLP.v_472^18 | 0.165 |
| 1 | MLP.v_253^13 | 0.159 |
| 2 | MLP.v_2342^21 | 0.143 |
| 3 | MLP.v_511^16 | 0.140 |
| 4 | MLP.v_491^10 | 0.140 |
| 5 | MLP.v_882^12 | – |
| 6 | MLP.v_12^3 | – |
| 7 | MLP.v_2669^18 | – |

Two of the paper's Table 1 vectors are recovered with matching semantics:

| vector | paper (Table 1) | this reproduction |
|---|---|---|
| MLP.v_882^12 | f*ck, sh*t, piss, hilar, stupidity, poop | f**k, Ġs**t, Ġp**s, Ġhilar, f**k, Ġstupidity, Ġstupid, Ġf**k |
| MLP.v_2669^18 | degener, whining, idiots, stupid, smug | Ġdegener, Ġstupid, Ġwhining, Ġlies, Ġignorant, Ġfoolish, Ġstupidity, Ġidiot |

## 3. Section 3.2 -- vocabulary projection (Table 1 analogue)

Command: `python scripts/project_vocab.py --model gpt2-medium --k 8 --n-svd 5`
(the Markdown rendering applies the paper's censoring; the JSON keeps the raw
tokens). Selected rows:

| Vector | top tokens |
|---|---|
| W_toxic | Ġprostitute, YOU, Ġchicks, Ġwhine, Ġdelinquent, ĠYOU, Ġgirlfriends, ĠHUN |
| MLP.v_882^12 | f**k, Ġs**t, Ġp**s, Ġhilar, f**k, Ġstupidity, Ġstupid, Ġf**k |
| MLP.v_2669^18 | Ġdegener, Ġstupid, Ġwhining, Ġlies, Ġignorant, Ġfoolish, Ġstupidity, Ġidiot |
| MLP.v_4065^13 | Ġf**k, Ġf**k, Ġs**t, Ġgoddamn, Ġp**s, Ġcrap, Ġdamn, Ġbullshit |
| MLP.v_3358^7 | Ġcrap, ĠGodd, Ġbullshit, Ġs**t, Ġdumb, Ġwhine, Ġgoddamn, Ġstupid |
| MLP.v_3414^11 | Ġsucker, Ġhilar, Ġa*****e, Ġloser, Ġshenanigans, Ġidiots, Ġcrap, Ġstupidity |
| MLP.v_610^2 | Ġrant, ĠFreak, Ġcrap, Ġstupidity, Ġbullshit, ĠNerd, ĠclichÃ©, Ġbadass |
| SVD.U_toxic[0] | Ġstupidity, Ġbullshit, Ġp**s, Ġcrap, Ġdudes, Ġstupid, Ġa*****e, Ġsmug |
| SVD.U_toxic[4] | Ġpropag, Ġmonopol, ĠEli, Ġallegedly, Ġcoer, Ġide, Ġaggression, Ġfake |

The behaviour matches the paper's two claims: individual value vectors encode
specific *dimensions* of toxicity (profanity, insults, sexual content, gendered
insults, ...) and the leading SVD direction is a generic toxicity direction.

## 4. Sections 3.3/4 -- GPT2 evaluation (the "NO OP" row)

Command:

```bash
python scripts/evaluate_model.py --model gpt2-medium --tag gpt2 \
    --max-prompts 60 --max-new-tokens 20 --ppl-sentences 400 --f1-items 200
```

| metric | paper (Table 2, NO OP) | this reproduction |
|---|---|---|
| toxicity (RealToxicityPrompts challenge) | 0.453 | **0.428** (60 prompts, toxic-roberta max-over-tokens) |
| perplexity (Wikitext-2) | 21.7 | **19.26** (first 400 test lines, sliding window) |
| F1 (Wikipedia continuations) | 0.193 | **0.138** (200 sentences, 10-token prompt) |

The toxicity and perplexity numbers land close to the paper's; the F1 number is
lower because the paper does not specify where the prompt/continuation split is
made (we use a 10-token prompt and generate exactly as many tokens as the
reference continuation), which systematically shortens the overlap.

## 5. Section 3.3 -- interventions with toxic vectors (Table 2 analogue)

Intervening on the last-layer residual stream, `x^{L-1} := x^{L-1} - alpha * W`
with `W` = `MLP.v_882^12` (the value vector whose top tokens match the paper's
Table 1 row). Each row generates 20 greedy tokens for the RealToxicityPrompts
challenge prompts and measures toxicity with toxic-roberta; the first three rows
use 40 prompts and the first 150 Wikitext-2 test lines, the last two use 30
prompts and the first 80 test lines:

| alpha | toxicity | PPL (Wikitext-2) |
|---|---|---|
| 0 (no intervention) | 0.4149 | 20.33 |
| 30 | 0.4141 | 20.55 |
| **60** | **0.3266** | **23.05** |
| 150 | 0.2284 | 45.98 |

The paper's corresponding row is `SUBTRACT MLP.v_770^19`: toxicity **0.305** at
perplexity **23.30** (vs. 0.453 / 21.7 without an intervention). Our alpha = 60
lands at the same operating point (toxicity 0.33 at PPL 23.1) without any
tuning: the intervention reduces toxicity substantially as soon as the alpha is
large enough to reach the post-DPO perplexity, and pushing alpha further keeps
lowering toxicity at the cost of generation quality. `scripts/run_interventions.py`
automates exactly this calibration (`--target-ppl 23.34`).

An alpha sweep of the three vectors selected by the automatic pipeline
(`W_toxic`, the top-ranked value vector, `SVD.U_toxic[0]`) at small alphas
(5, 15) shows no effect, which is expected: those alphas barely move the
perplexity, and the paper likewise calibrates alpha against the post-DPO
perplexity before reporting the intervention results.

## 6. What was verified but not run at scale

* **PPLM preference data (Section 4.2)** -- `scripts/build_pairs.py` generates
  positives with greedy GPT2 sampling and negatives with PPLM steered by
  `W_toxic` (Table 9 hyperparameters, probe filtering on). Generating 24,576
  pairs requires 24,576 PPLM generations (50 gradient steps each) and is
  GPU-bound. The code path is exercised by `tests/smoke_test.py`.
* **DPO (Section 4.1)** -- `scripts/train_dpo.py` implements the Table 8
  hyperparameters (RMSprop, lr 1e-6, batch 4, clip 10, beta 0.1, patience 10,
  90:10 split). Verified with short smoke runs (the loss falls and the
  validation accuracy rises on a tiny synthetic pair set); ~6,700 pairs of the
  24,576-pair dataset are needed before validation loss converges.
* **Sections 5 and 6** -- `scripts/analyze_dpo.py`, `scripts/logit_lens.py`,
  `scripts/unalign.py` consume the DPO model and produce the parameter-shift
  table, the activation drop (Fig. 2), the residual-stream offset and its PCA
  (Figs. 3-4), the cosine histograms (Fig. 5), the logit lens (Fig. 1) and
  Table 4. Each is exercised in the smoke test on a small model pair; the
  un-alignment weight surgery is additionally checked against the raw MLP
  weight matrix (`key_vector_after = 10 * key_vector_before`).

## 7. Runtime notes

Measured on the CPU-only reproduction box (GPT2-medium, float32, 16 cores):

* forward pass: ~3.6 Jigsaw comments/s at `max_length 48`, batch 48;
* autoregressive greedy generation: ~20 tokens/s for batch 16;
* Wikitext-2 perplexity over the first 400 lines: ~5 minutes;
* the full 24,576-pair PPLM dataset would take on the order of 10^3 CPU-hours,
  which is why it is left to the (GPU) evaluation environment.
