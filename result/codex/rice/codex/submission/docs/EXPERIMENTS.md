# Experiment-by-experiment reproduction notes

All numbers quoted as "paper" are the values reported in the ICML 2024 paper.
Every experiment writes one JSON file per application into `results/`; the
plotting script turns them into the figures of the paper.

## Experiment I — fidelity and efficiency of the explanation

*Command*: `scripts/run_experiment1_fidelity.py --env <ENV>`

* Fidelity of `Random`, `StateMask` and `Ours`, 500 trajectories, 3 seeds,
  `K ∈ {10, 20, 30, 40}%` → `results/experiment_i/<ENV>_fidelity.json`,
  figure `fidelity_<ENV>.png` (paper Figure 5).
* Mask training time for a *fixed number of samples* (Table 4 of the paper:
  MuJoCo `3·10^5`, selfish mining `1.5·10^6`, CAGE `10^7`, driving `2443260`).
  The paper measures an average 16.8 % drop of the training time with our
  objective, because StateMask additionally estimates the discounted
  accumulated rewards of the perturbed and of the target agent.

Paper behaviour to reproduce: **ours ≈ StateMask ≫ Random** in fidelity, and
our mask network trains faster at an equal sample budget.

### Keeping the mask network in the non-degenerate regime

`R'(s,a) = R(s,a) + alpha * a^m` has two corner solutions: never blind the agent
(when `alpha` is negligible next to the reward scale, which makes every
importance score ≈ 1) and always blind it (when `alpha` dominates, which makes
every importance score ≈ 0).  In both corners the explanation carries no
information and the fidelity metric degenerates to a fixed window.

The paper relies on the Stable-Baselines3 pipeline, which normalises the reward
by the running standard deviation of the returns, so `alpha = 0.01` is a
one-percent perturbation of a step reward.  `EnvSpec.mask_reward_scale = None`
reproduces that behaviour (`1.0` gives the raw-reward formula).  Measured on the
selfish mining application of this repository (pre-trained PPO agent, 20k mask
samples, fidelity over 25 trajectories):

| setting | mask rate | importance spread | fidelity (K=10 %, K=30 %) |
|---|---|---|---|
| raw reward, `alpha=0.01` | 0.01 | 0.007 | 0.44 / 0.36 |
| raw reward, `alpha=0.05` | 0.01 | 0.008 | – |
| normalised, `alpha=0.01` | 0.00 | 0.001 | – |
| normalised, `alpha=0.05` | 0.64 | 0.85 | **2.32 / 1.31** |
| random explanation | – | – | 1.65 / 0.87 |

i.e. as soon as the mask network is in the interior regime, the fidelity of our
explanation exceeds the random baseline at every `K` — the trend claimed by
Experiment I.  `EnvSpec.alpha` therefore defaults to the paper's 0.01 and is
raised to 0.05 for the simplified selfish mining MDP (whose reward scale differs
from the original blockchain simulator).

`scripts/calibrate_alpha.py` automates the choice: it trains short mask networks
for `alpha in {0.001, 0.01, 0.02, 0.05, 0.1}` and reports the candidate whose
mask rate is closest to 0.5.  On the selfish mining agent used above it returns

| alpha | 0.001 | 0.01 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|---|
| mask rate | 0.05 | 0.07 | 0.17 | **0.54** | 0.58 |

which is the monotone control of the mask ratio that the paper attributes to
`alpha`, and confirms the per-application value used here.

## Experiment II — effectiveness of the refining method

*Command*: `scripts/run_experiment2_refine.py --env <ENV> --seeds 0 1 2`

Same explanation (ours) for every refining method; final reward after refining
(dense-reward applications) or the performance *during* refining (sparse
applications, Figure 2).

Paper Table 1 (left half) / Figure 2 — see the reference table in the README.
The key qualitative facts: RICE improves the pre-trained agent the most, PPO
fine-tuning barely moves, StateMask-R can degrade the agent (Cage Challenge 2:
−23.64 → −26.98).

## Experiment III — refining with different explanations

*Command*: `scripts/run_experiment3_explanations.py --env <ENV>` (also part of
`run_experiment2_refine.py`, right half of Table 1)

RICE is fixed and the explanation varies.  Paper: Ours ≥ StateMask > Random.
The addendum notes that the "Ours vs StateMask" ordering in this table is not a
significant claim (they are comparable), so only "mask-based ≫ random" is
expected to hold robustly.

## Experiment IV — refining an agent trained by another algorithm

*Command*: `scripts/run_experiment4_sac.py --env Hopper`

1. Pre-train SAC in Hopper (Stable-Baselines3 defaults).
2. GAIL imitation of the SAC agent to obtain a PPO-compatible policy
   (`rice/imitation.py`).
3. Refine the imitated policy with RICE, PPO fine-tuning, StateMask-R, JSRL and
   SAC fine-tuning.

Paper Figure 3: RICE reaches the highest reward, SAC fine-tuning stays at the
bottleneck, and switching to PPO-based refining is what unlocks the improvement.
Figure 6 (sensitivity of `p` and `λ` for the imitated agent) is produced by
`rice/experiments/hyperparams.run_p_lambda_grid` when it is given the GAIL
checkpoint instead of the PPO checkpoint.

## Experiment V — hyper-parameters

*Command*: `scripts/run_experiment5_hyperparams.py --env <ENV>`

* `p ∈ {0, 0.25, 0.5, 0.75, 1}` and `λ ∈ {0, 0.1, 0.01, 0.001}`
  (`run_p_lambda_grid`) → `results/experiment_v/<ENV>_p_lambda.json`,
  figure `p_lambda_<ENV>.png` (paper Figures 7 and 8; the addendum clarifies
  that Figure 7 sweeps `λ` and Figure 8 sweeps `p`).
* `α ∈ {0.01, 0.001, 0.0001}`: retrain the mask network for each value and
  report fidelity → `results/experiment_v/<ENV>_alpha.json`,
  figure `alpha_<ENV>.png` (paper Figure 9).

Paper behaviour: `p = 0` and `p = 1` are worse than `0 < p < 1` (the mixed
initial state distribution matters), any `λ > 0` improves over `λ = 0` and the
method is not sensitive to the exact `λ`, and fidelity barely changes with `α`.
The hyper-parameter results of the *sparse* games are out of scope.

## Appendix-only experiments that are not reproduced

| Item | Status |
|---|---|
| Table 5 (Self-Imitation Learning) | out of scope |
| Table 6 (Integrated Gradients, AIRS) | out of scope |
| Figure 10 (SparseWalker2d refining) | out of scope |
| Figures 11-13 (sparse hyper-parameters) | out of scope |
| Figures 14-15 (MountainCar, driving visualisation) | out of scope |
| Appendix D (Malware Mutation case study) | out of scope |
| Section 3.4 (sub-optimality bound) | out of scope |

## Runtime expectations

The paper trained on 8×A100 GPUs.  On a single CPU core this repository runs
roughly 160-700 environment steps per second for the MuJoCo games, so the full
budgets (`3·10^5` refining steps per method and seed, `500` fidelity
trajectories) take hours; use the reduced budgets of the smoke tests
(`tests/`) for a quick end-to-end validation and the full budgets for the
reported numbers.
