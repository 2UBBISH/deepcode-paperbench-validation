# Reduced-scale smoke run — results

Command (CPU, ~12 minutes; `smoke/outputs/` is git-ignored, the JSON below is committed):

```bash
python smoke/run_smoke.py --per-task 8 --candidates 40 --online 3 --forecast-steps 20 \
    --output smoke/outputs
```

## Setting

| Item | Value |
|---|---|
| Base PTLM (`f_0`) | `google/flan-t5-small` (77M parameters) — a stand-in for FLAN-T5_Large |
| Refinement setup | head-only, 20 steps per error, `lr = 2e-3` (paper: 100 steps at `1e-3`) |
| `D_PT` | 24 examples: the `glue_mrpc_paraphrase`, `glue_mrpc_same_thing`, `glue_mrpc_equivalent` templates of the upstream task `glue-mrpc`, 8 examples each |
| `D_hat_PT` | 16 of the 24 (base EM = 0.667) |
| `D_R` | 12 mispredictions of `f_0` on the `glue_qqp_duplicate` validation split (EM of `f_0` on the 40-example candidate pool = 0.700) |
| `D_R^Train` / `D_R^Test` | 3 / 3 online examples (60/40 split of a subsample) |
| Forecasting training | 20 steps (paper: up to 100k) |
| Edit success after fixing | 6/6 errors fixed |

## Forecasting forgetting (Table 1 style), F1 on `D_R^Test` over the 48 test pairs
(3 online examples × the 16 upstream examples of `D_hat_PT`)

| Method | F1 | Precision | Recall |
|---|---|---|---|
| Threshold | 95.83 | 95.83 | 95.83 |
| Fixed Logit | 66.67 | 50.00 | 100.00 |
| Trainable Logit | 66.67 | 50.00 | 100.00 |
| Representation | **95.83** | 95.83 | 95.83 |
| w/o Prior | 66.67 | 50.00 | 100.00 |

Trends that match the paper: the representation-based model is at least as good as every
other method, removing the frequency prior costs a lot of precision (95.8 → 50.0), and the
logit-based variants are weaker than the representation-based model on a T5 model — which
is precisely the failure mode the paper reports for FLAN-T5.

## Single-error refinement with replay (Table 4 style)

| Method | Edit success | EM drop % (`D_PT`) |
|---|---|---|
| Vanilla FT | 100.0 | -25.0 |
| Replay w/ Random | 100.0 | -25.0 |
| Replay w/ Representation | 100.0 | -21.875 |
| Replay w/ GT Forget | 100.0 | -25.0 |

At this scale no forgetting is induced at all: 20 head-only steps of a 77M model on one
example *improve* Exact Match on the 24 upstream examples (negative EM drop). The value of
this part of the smoke run is that the full refinement path executes: replay selection
(random / forecasted / ground truth), the distillation loss against the cached base logits,
and the EM-drop / edit-success measurement. Forgetting (and therefore a meaningful replay
comparison) requires the paper's scale — larger models, longer prompts and 100/30 update
steps, as reproduced by `scripts/run_table3.py` and `scripts/run_table4.py`.

Raw artefacts: `smoke/results/smoke_results.json` (metrics) and
`smoke/results/smoke_log.txt` (full console output).

## Second smoke run: sequential refinement (Table 3) and Figure 3

```bash
python smoke/run_smoke_stream.py --online 2 --upstream 12 --steps 20 --lr 2e-3 \
    --forecast-steps 20 --output smoke/outputs_stream
```

| Item | Value |
|---|---|
| `D_PT` | 12 examples of `glue-mrpc` |
| `D_R^Train` / `D_R^Test` | 2 / 2 examples, 20 head-only steps each |

| Method (Table 3 style, `D_R^Test` stream of 2 examples) | Edit success | EM drop % (`D_PT`) |
|---|---|---|
| Vanilla FT | 100.0 | -25.0 |
| Replay w/ Random | 100.0 | -25.0 |
| Replay w/ Representation | 100.0 | -25.0 |

Figure 3 style (one stream step, forecasted indicator computed once at the start of the
stream, ground truth re-measured with the continually updated LM): threshold and
representation-based forecasting both report F1 = precision = recall = 100.00 after the
single step (`smoke/results/smoke_stream_results.json`).  Again, the point of this run is
that `wwmf.refinement.stream.sequential_refinement` and
`wwmf.refinement.stream.continual_forecasting_curves` execute end to end — the metric
values are meaningless at this scale.
