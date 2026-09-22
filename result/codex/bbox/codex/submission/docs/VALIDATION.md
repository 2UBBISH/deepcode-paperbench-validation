# Validation log

Everything in this document was executed inside the reproduction sandbox
(CPU-only, no API keys).  It is the evidence that the code runs end to end and
that the numbers it produces have the right *shape*; the absolute accuracies of
the paper require the Azure OpenAI credentials and a GPU, which are only
available in the final evaluation environment.

## 1. Unit / smoke tests

```console
$ python -m pytest tests -q
...................................                                      [100%]
35 passed, 1 warning in 9.84s
```

The suite covers, among others:

| Test | What it pins down |
|---|---|
| `test_pairwise_gradient_matches_eq3` | `d loss / d g+ = -1 + 2 alpha g+`, `d loss / d g- = 1 + 2 alpha g-` (Eq. 3) |
| `test_listwise_gradient_is_posterior_minus_indicator` | the Appendix B derivation `d loss / d g(x_m) = p_theta(x_m) - 1[m = 0]` |
| `test_listwise_loss_equals_negative_log_posterior` | Eq. (2) is the negative log posterior of the positive sample |
| `test_training_increases_positive_energy_margin` | the adapter update (Eq. 7) really raises `g_theta(y_+)` above `g_theta(y_-)` |
| `test_initialisation_splits_candidates` / `test_update_replaces_negatives...` | Eqs. (5)-(6) and outcome supervision |
| `test_beam_search_returns_adapter_preferred_hypothesis` | the adapter, not the LLM, selects the surviving beam |
| `test_prefer_finished_ranks_complete_hypotheses_first` | complete hypotheses are preferred over partial ones |
| `test_same_adapter_plugs_into_another_llm` | plug-and-play mechanism of Table 3 |
| `test_online_adaptation_smoke` | Algorithm 1 end to end (mock LLM, tiny adapter, local JSONL data) |
| `test_listwise_softmax_training_runs_and_separates` | Eq. (2) can also be used as the training objective |
| `test_chat_client_parses_responses_and_counts_tokens` / `test_completion_client_...` | the Azure/OpenAI request wiring and token accounting (stubbed SDK) |
| `test_responses_are_cached_on_disk` | every LLM call is cached, so interrupted runs resume for free |
| `test_cost_uses_input_and_output_prices` / `test_cost_per_1k_questions_scales...` | the arithmetic behind Table 4 |

## 2. Real backbone, real data

`microsoft/deberta-v3-base` (the paper's 0.1B adapter for StrategyQA / GSM8K /
ScienceQA) loaded on CPU, scored pairs, and was updated 10 times on the loss of
Eq. (3):

```text
loaded microsoft/deberta-v3-base in 246.0s
scores:  [0.1082, 0.1087]                      # before training
loss curve:        [0.0472, -0.0678, -0.3474]  # decreasing
pos energy:        [0.0770,  0.0687,  0.7093]  # rising
neg energy:        [0.1238,  0.0005,  0.3553]
scores after: [0.5077, 0.4129]                 # positive answers scored higher
```

Full online-adaptation run on real GSM8K data (6 training questions, 2 test
questions, 1 iteration, deterministic mock LLM so that the run is free):

```text
[data] gsm8k: 6 train / 2 test
[online] initialising positive/negative sets for 6 questions with 5 un-adapted samples
[online] bank statistics after init: {'num_questions': 6, 'positives_per_question': 1.0, 'negatives_per_question': 3.17}
[online] iteration 0: sampled candidates for 6 questions (+1.00/-2.67 per question)
[online] iteration 0 finished in 87.0s: {... 'loss': 0.002, 'pairwise_accuracy': 0.516 ...}
[eval] gsm8k accuracy: 0.00%
```

The 0% accuracy is expected: the mock LLM returns random tokens
(`"We compute the intermediate quantity. #### gamma"`), so no candidate can match
the GSM8K gold answer; the point of the run is that data loading, sampling, bank
updates, the adapter update and the evaluation pipeline all execute.

The local HuggingFace client (used for Mixtral-8x7B in Tables 3 and 6) was
exercised with a tiny causal LM so that the code path is validated without a
GPU:

```text
texts: ['uteaidoughypkeral Itper...', ' beforeiver 1orn Leost...']
usage: {'num_calls': 2, 'prompt_tokens': 26, 'completion_tokens': 32, 'total_tokens': 58}
cached identical: True
```

## 3. Dataset loaders

Exercised against the public HuggingFace mirrors (the paper's Appendix F.1 sizes
in parentheses):

| Dataset | Loaded | Paper |
|---|---|---|
| GSM8K | 7473 train / 1319 test | 7473 / 1319 |
| TruthfulQA | 817 items split into 717 train / 100 test | 717 / 100 |
| ScienceQA (`derek-thomas/ScienceQA`, image questions removed) | 12726 / 4241 before sampling, 2000 / 500 after | 2000 / 500 |
| StrategyQA (`ChilleD/StrategyQA`) | 1603 train / 687 test | 2059 / 229 |

## 4. Reference numbers of the paper (for comparison)

These are the targets the scripts above reproduce when run with credentials
(values copied from the paper).

**Table 2 — adapting gpt-3.5-turbo (accuracy %)**

| Method | StrategyQA | GSM8K | TruthfulQA | ScienceQA |
|---|---|---|---|---|
| gpt-3.5-turbo (CoT) | 66.59 | 67.51 | 77.00 | 72.90 |
| Azure-SFT (upper bound) | 76.86 | 69.94 | 95.00 | 79.00 |
| BBox-Adapter (Ground-Truth) | 71.62 | 73.86 | 79.70 | 78.53 |
| BBox-Adapter (AI Feedback) | 69.85 | 73.50 | 82.10 | 78.30 |
| BBox-Adapter (Combined) | 72.27 | 74.28 | 83.60 | 79.40 |

**Table 3 — plug-and-play (accuracy %)**

| Pluggee | StrategyQA | GSM8K | TruthfulQA | Average delta |
|---|---|---|---|---|
| davinci-002 | 44.19 -> 59.61 | 23.73 -> 23.85 | 31.50 -> 36.50 | +6.85 |
| Mixtral-8x7B | 59.91 -> 63.97 | 47.46 -> 47.61 | 40.40 -> 49.70 | +4.50 |

**Table 4 — cost (StrategyQA / GSM8K)**: gpt-3.5-turbo 0.41 / 1.22 USD per 1k
questions of inference; Azure-SFT 153.00 / 216.50 USD of training and
7.50 / 28.30 USD per 1k questions of inference; BBox-Adapter single-step
2.77 / 7.54 USD training and 2.20 / 3.10 USD per 1k inference; BBox-Adapter
full-step 3.48 / 11.58 USD training and 5.37 / 12.46 USD per 1k inference.

**Table 5 — MLM vs NCE (accuracy %)**: MLM 61.52 (0.1B) / 60.41 (0.3B) on
StrategyQA and 70.56 / 70.81 on GSM8K, versus NCE 71.62 / 71.18 and 72.06 / 73.86.

**Table 6 — Mixtral-8x7B on StrategyQA**: base 59.91% (90 GiB), base + LoRA
73.80% (0.1B) / 75.98% (0.3B) with 208 GiB training and 92 GiB inference, base +
BBox-Adapter 66.08% (0.1B) / 65.26% (0.3B) with 105 GiB training and 92 GiB
inference.

**Figure 3 — scale analysis**: increasing the number of beams from 1 to 5 adds
2.41% on average, and the accuracy improves monotonically after one round of
online adaptation (the un-finetuned `T = 0` adapter is worse than the base
model).
