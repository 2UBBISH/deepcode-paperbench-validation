# Paper equations and algorithms → code

Line-by-line map of the paper's formal parts to this repository. File references are
relative to the repository root.

## Metrics (Sec. 2)

| Paper | Code |
|---|---|
| `EM_{D,f} = |{<x,y> in D : f(x)=y}| / |D|`, graded with the SQuAD 2.0 script | `wwmf/evaluation/metrics.py::exact_match`, `normalize_answer` |
| Edit Success Rate | `wwmf/evaluation/metrics.py::edit_success_rate` |
| EM Drop Ratio `(EM_{D_PT,f_i} - EM_{D_PT,f_0}) / EM_{D_PT,f_0}` | `wwmf/evaluation/metrics.py::em_drop_ratio` |
| Forgetting label `z_ij = 1[f_i(x_j) != y_j]` on `D_hat_PT` | `wwmf/forecasting/cache.py::build_online_artifacts` (labels), `ForecastContext.eval_mask` (the `D_hat_PT` mask) |

## Sec. 3.1 — frequency-threshold forecasting

| Paper | Code |
|---|---|
| `g(<x_i,y_i>,<x_j,y_j>) = 1[|{i in D_R^Train : z_ij = 1}| >= gamma]` | `wwmf/forecasting/threshold.py::ThresholdForecaster.predict` (`counts >= gamma`) |
| `gamma` tuned to maximise the F1 on `D_R^Train` | `ThresholdForecaster._tune_gamma` (exact F1 maximisation, verified against brute force in `tests/test_forecasting.py`) |

## Sec. 3.2 — logit-change forecasting

| Paper | Code |
|---|---|
| `Delta f_i(x_i) = f_i(x_i) - f_0(x_i)` | `wwmf/forecasting/cache.py::build_online_artifacts` (`delta_logits`, reduced to the candidate vocabulary) |
| `Theta_tilde(x_j,x_i) = h(x_j,y_j) h(x_i,y_i)^T` with `h: (x,y) -> R^{T x d}` | `TrainableLogitForecaster.fit`: `torch.einsum("btd,bid->bti", H_j, H_i)` in `wwmf/forecasting/logit_change.py` |
| `f_hat_i(x_j) = Theta_tilde f_i(x_i)-f_0(x_i)] + f_0(x_j)` | `_LogitForecasterBase._predicted_logits` (+ `predict_forgetting_from_logits` for `arg max != y_j`) |
| Margin loss Eq. 3 | `wwmf/forecasting/logit_change.py::margin_loss` (sign `(-1)^z`, down-weighting of positives `alpha=0.1`) |
| Fixed (untrained) variant: `h` = frozen final-layer representation of the base PTLM | `FixedLogitForecaster` (`wwmf/forecasting/logit_change.py`); the frozen online representation is attached in `build_online_artifacts(..., cache_frozen_reps=True)` |
| "we only cache top k = 100 largest logits for each token in y_j" | `wwmf/forecasting/cache.py::build_candidate_vocab` + `CandidateVocab` |
| Algorithm 1 (training) | `TrainableLogitForecaster.fit` — 16 pairs per batch (8 positive / 8 negative), LM LR `1e-5`, MLP LR `1e-4` |
| Algorithm 2 (inference) | `TrainableLogitForecaster.predict` — one forward pass over `D_PT` caches `h(x_j,y_j)`, then one kernel product per online example (no LM inference on `D_PT`) |

## Sec. 3.3 — representation-based forecasting

| Paper | Code |
|---|---|
| `g = sigma(h(x_j,y_j) h(x_i,y_i)^T + b_j)` | `wwmf/forecasting/representation.py::RepresentationForecaster.scores` (`upstream_reps @ h_i + prior`) |
| Averaged representation of all tokens in `<x,y>` | `wwmf/forecasting/encoders.py::PairEncoder.forward` (`pooled_dec`, `pooled_enc`) |
| Frequency prior `b_j = log p(z_j=1) - log p(z_j=0)` over `D_R^Train` | `wwmf/forecasting/base.py::frequency_prior`, called in `RepresentationForecaster.fit` |
| BCE loss, `alpha=0.1` on positives | `RepresentationForecaster.fit` (`binary_cross_entropy_with_logits(..., weight=...)`) |
| "w/o Prior" ablation (Tables 1 and 2) | `RepresentationForecaster(use_prior=False)` |
| Algorithm 3 / 4 | `RepresentationForecaster.fit` / `.predict` |

## Sec. 4.1 / 4.2 — setups and compared methods

| Paper | Code |
|---|---|
| Head-only / LoRA / full FT, 100 : 30 : 30 steps | `wwmf/models/tuning.py::mark_trainable`, `fix_single_error`, `wwmf/config.py` |
| LoRA on query/value matrices, `r=16`, `alpha=32`, dropout 0.1 | `wwmf/models/tuning.py::apply_lora` (`LORA_CONFIG`) |
| Replay with a distillation loss against the base PTLM, 8 examples every 10 steps (4 every 5 on 3B) | `wwmf/refinement/replay.py::ReplayPool`, `make_replay_callback`; `wwmf/models/tuning.py::distillation_loss` (MSE on pre-softmax logits, restricted to the cached candidate vocabulary) |
| Replay of examples predicted to be forgotten / ground-truth forgotten | `ScoreReplay`, `GTForgetReplay` (`wwmf/refinement/replay.py`) |

## Sec. 5.2 / 5.3 — experiments

| Paper | Code |
|---|---|
| Table 1 (F1 of forecasting, one error at a time) | `wwmf/experiments.py::run_forecasting_experiment`, `scripts/run_table1.py` |
| Table 2 (in-domain / out-of-domain, BART0) | `run_ood_experiment`, `wwmf/data/build.py::build_ood_split`, `scripts/run_table2.py` |
| Table 3 (sequential refinement + replay) | `wwmf/refinement/stream.py::sequential_refinement`, `scripts/run_table3.py` |
| Table 4 (single errors, separately) | `evaluate_single_error_replay`, `scripts/run_table4.py` |
| Figure 3 (F1 / precision / recall over the stream) | `continual_forecasting_curves` (prediction frozen at the stream start, `z_ij^t` re-measured each step), `scripts/run_figure3.py` |
| Table 5 / Sec. 5.3 (complexity) | `wwmf/analysis/complexity.py`, `scripts/run_complexity.py` |
| Figure 2(a) (logit-change transfer) | `wwmf/analysis/logit_transfer.py`, `scripts/analyze_logit_transfer.py` |
| Appendix B (forecasting training details) | `wwmf/config.py::FORECAST_TRAIN` |
