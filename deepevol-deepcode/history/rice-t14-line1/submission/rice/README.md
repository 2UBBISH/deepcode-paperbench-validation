# RICE — A Refining scheme for ReInforcement learning with Explanation

Reproduction of **RICE** (Proc. 41st ICML, PMLR 235, 2024), reference code
<https://github.com/chengzelei/RICE>.

RICE identifies *critical states* ("exploration frontiers") in a warm-started,
bottlenecked DRL policy with a re-designed **StateMask** mask network, builds a
**MIXED initial-state distribution** from the default initial-state distribution
and the identified critical states, and refines the policy with **PPO +
Random-Network-Distillation (RND)** exploration bonus — without retraining from
scratch.

> **Scope note (benchmark addendum).** This repository reproduces the paper's
> **TRENDS and takeaways**, not exact numbers. Comparisons are qualitative
> (`ours ≈ StateMask > Random`, `ours ≳ baselines`, monotone sensitivity
> structure). Explicit out-of-scope items are listed at the end.

---

## 1. What is implemented

| Paper element | Where |
|---|---|
| **Algorithm 1** — mask-network training (vanilla PPO + `alpha` blinding bonus) | `rice/algorithms/mask_network.py` |
| Masking rule Eq. (1) `a_t ⊙ a_t^m`; importance `= P(a_t^m = 0 \| s_t)`; `argmax` critical state | `rice/algorithms/mask_network.py`, `rice/algorithms/critical_state.py` |
| **Mixed initial state** `mu(s) = beta d_rho^{pi_hat}(s) + (1-beta) rho(s)`, realised as the Bernoulli(`p`) roll-in of Algorithm 2 | `rice/algorithms/mixed_init.py` |
| **RND** bonus `R^RND = ||f(s_{t+1}) - f_hat(s_{t+1})||^2` with normalisation; `R' = R + lambda R^RND` | `rice/algorithms/rnd.py` |
| **Algorithm 2** — the RICE refining loop | `rice/algorithms/refine.py` (+ shared PPO in `rice/algorithms/ppo.py`) |
| **Go-Explore style reset** to a critical state (Ecoffet et al. 2019, §C.1) | `rice/algorithms/env_reset.py` |
| **Fidelity Score** `log(d/d_max) - log(l/L)`, `l = L*K` (Experiment I) | `rice/evaluation/fidelity_score.py` |
| Refining evaluation (Experiments II/III/IV), Table 1/5 trend checks | `rice/evaluation/refining_eval.py` |
| Hyper-parameter study (Experiment V: `p`, `lambda`, `alpha`) | `rice/evaluation/hyperparam_sweep.py` |
| Environments: Hopper-v3, Walker2d-v3, Reacher-v2, HalfCheetah-v3, SparseHopper, SparseHalfCheetah, SelfishMining, CageChallenge2, Macro-v1 | `rice/environments/*` |
| Explanations: **ours** (mask net), StateMask, Random, Integrated Gradients, AIRS | `rice/explanation/*` |
| Baselines: PPO fine-tuning, StateMask-R, JSRL, SIL, SAC(+GAIL) | `rice/baselines/*` |
| Warm-start pre-training (`pi`) for all tasks | `rice/scripts/pretrain_agent.py` |

Each algorithm module is a small, self-contained implementation written from the
paper's equations (Eq. (1), Eqs. in §3.3, Algorithm 1, Algorithm 2, the fidelity
metric of §4.1). See the module docstrings for the equation-by-equation mapping.

---

## 2. Installation

Python **3.8+** (3.9/3.10 recommended).

```bash
cd rice
pip install -r requirements.txt
```

`requirements.txt` pins the core stack (`numpy`, `torch`, `matplotlib`) and the
environment stack (`gym`, `gymnasium`, `mujoco`). **The paper specifies no exact
package versions**; the pins in `requirements.txt` are the versions this
reproduction was developed against. Optional heavy back-ends are commented out
and are *not* required — every module degrades gracefully to an in-repo
implementation when they are absent:

| Optional dependency | Env var to enable it | Fallback used instead |
|---|---|---|
| `cage-challenge-2` (Cardiff champion) | `RICE_USE_REAL_CAGE=1` | in-repo pure-Python CAGE-2 simulator |
| `metadrive-simulator` / DI-drive | `RICE_ALLOW_METADRIVE=1` | in-repo Numpy Macro-v1 driving simulator |
| upstream `RL-state_mask` | — | faithful local `MaskNetwork` (Algorithm 1) |
| upstream AIRS | `RICE_AIRS_UPSTREAM=1` | attention / gradient / occlusion attribution |
| MuJoCo (`mujoco` not importable) | `RICE_ALLOW_MUJOCO_FALLBACK=1` | analytic locomotion stand-in (CPU smoke tests only) |

SAC + GAIL (Experiment IV) is additionally gated behind `RICE_ENABLE_SAC=1`
because it costs ≈1e6 (SAC) + 3e5 (GAIL) environment steps.

---

## 3. Layout

```
rice/
├── rice/
│   ├── algorithms/      mask_network, critical_state, mixed_init, rnd, refine, ppo, env_reset
│   ├── environments/    mujoco_dense, mujoco_sparse, selfish_mining, cage_challenge2, autodriving
│   ├── explanation/     random_explanation, statemask_adapter, integrated_gradients, airs_adapter
│   ├── baselines/       ppo_finetune, statemask_r, jsrl, self_imitation, sac_gail
│   ├── evaluation/      fidelity_score, refining_eval, hyperparam_sweep
│   ├── configs/         one YAML per environment (p, lambda, alpha from Table 3)
│   ├── utils/           seeding, buffers, normalization, logging, plotting, io
│   └── tests/           test_mask_network, test_mixed_init, test_rnd, test_fidelity_score, test_env_reset
├── scripts/             pretrain_agent, train_mask, fidelity_eval, run_refine, run_baselines,
│                        run_sac_gail, run_all_experiments.sh
├── main.py              single CLI entry point
├── requirements.txt
└── README.md
```

Both the nested (`rice/rice/...`) and flat (`rice/...`) import layouts are
supported: scripts bootstrap `sys.path` themselves, so the commands below work
from the repository root, from `rice/`, or against an installed package.

---

## 4. Running the experiments

### 4.0 Single entry point

```bash
python main.py list            # tasks, methods, explanations, registered envs
python main.py test            # run the pytest suite
python main.py --help
```

Sub-commands: `pretrain`, `train-mask`, `fidelity`, `refine`, `baselines`,
`sac-gail`, `sweep`, `all`, `list`, `test` (aliases such as `mask`, `exp1`,
`sensitivity`, `pytest` are accepted).

### 4.1 Stage 0 — warm-start (bottlenecked) policies `pi`

Train each agent until its performance plateaus near the paper's "No Refine"
level (Table 1), then checkpoint. Plateaus are detected with a target ratio of
the reference return plus patience (`--target-ratio`, `--patience`).

```bash
python main.py pretrain --task Hopper-v3 --backend sb3 \
    --total-timesteps 1000000 --target-ratio 0.95 --seeds 0 1 2 \
    --out-dir runs/pretrain
```

Weights land in `runs/pretrain/weights/{task}_seed{seed}.pt` and are consumed by
the refiner via `rice.algorithms.refine.load_policy_weights`.
Use `--backend rice` to pre-train with the in-repo PPO (no Stable-Baselines3
needed, works on CPU); `--backend sac` feeds Experiment IV.

### 4.2 Stage 1 / Experiment I(a) — mask network (Algorithm 1, Table 4 timing)

```bash
python main.py train-mask --task Hopper-v3 --seeds 0 1 2 \
    --weights runs/pretrain/weights --out-dir runs/mask
```

* The **sample budget is fixed per task** from Table 4 (see §5 below); outer
  iterations are derived as `budget / steps_per_iter`.
* `alpha` is taken from Table 3 (**1e-4**), see the conflict note in §6.
* Artifacts: `mask_{task}.json/.csv`, `table4_{task}.png` (wall-clock vs the
  StateMask reference), `weights/{task}_seed{seed}.pt`, plus a
  `mean_mask_rate` per run as the **anti-collapse check** (a mask that always
  outputs `0` is flagged as collapsed).

### 4.3 Experiment I(b) — fidelity of the explanation

500 trajectories × `K ∈ {10, 20, 30, 40}%` × 3 seeds, mean ± std.

```bash
python main.py fidelity --task Hopper-v3 --explanations ours statemask random \
    --trajectories 500 --seeds 0 1 2 --mask-weights runs/mask/weights \
    --out-dir runs/fidelity
```

Pipeline per trajectory (per the addendum): roll out `pi` for `L` steps → slide a
window of width `l = L*K` → keep the **highest average importance** segment
(tie → earliest) → fast-forward to it → replace actions inside the window with
random actions (Eq. (1)'s `a_random`) → resume `pi` → measure
`d = |R' - R|` → `score = log(d/d_max) - log(l/L)`.

`d_max` is **not specified in the paper** and is therefore estimated from warm-up
episodes (`--d-max` overrides it). Outputs `fidelity_{task}.json/.csv` and
`fidelity_scores.png` (Figure 5 style), with `notes_{task}.txt` recording the
`d_max` estimate.

### 4.4 Experiments II / III / IV — refining

```bash
# Experiment II: fix explanation to ours, vary the refining method
python main.py refine --task Hopper-v3 --methods no_refine ours ppo statemask_r jsrl \
    --weights runs/pretrain/weights --mask-weights runs/mask/weights --seeds 0 1 2

# Experiment III: fix refining to ours, vary the explanation
python main.py refine --task Hopper-v3 --methods ours \
    --explanations ours statemask random --weights runs/pretrain/weights --mask-weights runs/mask/weights

# Experiment IV: SAC pre-train -> GAIL imitation -> refine (needs RICE_ENABLE_SAC=1)
RICE_ENABLE_SAC=1 python main.py sac-gail --task Hopper-v3 \
    --methods ours ppo statemask_r jsrl sac --seeds 0 1 2 --out-dir runs/sac_gail
```

`run_baselines.py` is the equivalent driver with per-baseline dispatch
(`--methods ppo statemask_r jsrl sil sac_gail`). Sparse tasks automatically
produce **refining curves** (Figure 2 style) in addition to final rewards
(Figure 3 / Table 1 style).

Method → configuration mapping (all share the same PPO update, so RICE differs
from the baselines *only* in mixed-init and RND):

| Method | `p` | `lambda` | Notes |
|---|---|---|---|
| `ours` (RICE) | Table 3 | Table 3 | Bernoulli roll-in + RND bonus |
| `ppo` (fine-tuning) | 0 | 0 | lowered learning rate, continue PPO |
| `statemask_r` | 1 | 0 | always reset to the critical state |
| `jsrl` | 1 | 0 | `pi_e` initialised to `pi_g`, annealed guided horizon |
| `sil` | 0 | 0 | past-good-experience replay loss |
| `sac` | — | — | SAC continued training (Experiment IV) |
| `no_refine` | — | — | evaluation of the warm-start policy only |

### 4.5 Experiment V — hyper-parameter sensitivity

```bash
python main.py sweep --param p      --task Hopper-v3 --seeds 0 1 2
python main.py sweep --param lambda --task SelfishMining
python main.py sweep --param alpha  --task Hopper-v3      # fidelity mode by default
```

Grids (from §4.2):

* `p ∈ {0, 0.25, 0.5, 0.75, 1}` — refining mode
* `lambda ∈ {0, 0.1, 0.01, 0.001}` — refining mode
* `alpha ∈ {0.01, 0.001, 0.0001}` — fidelity mode by default (`--mode refine` to override)

Outputs `sweep_{param}_{task}.json/.csv` and Figure 7/8/9-style PNGs with the
`p ∈ [0.25, 0.5]` band shaded.

### 4.6 Everything at once

```bash
bash scripts/run_all_experiments.sh                 # pretrain, mask, exp1, exp2, exp3, exp5, tests
bash scripts/run_all_experiments.sh --quick         # smaller budgets
bash scripts/run_all_experiments.sh --smoke --dry-run
bash scripts/run_all_experiments.sh --enable-sac    # adds Experiment IV
bash scripts/run_all_experiments.sh --experiments mask exp1 --tasks Hopper-v3 Reacher-v2
```

The orchestrator writes `run_all_experiments_summary.json/.md`, a per-stage TSV
and a log into `--out-dir` (default `runs/all`).

---

## 5. Hyper-parameters (Table 3 / Table 4)

Table 3 (`rice/configs/*.yaml`, `TABLE3_HYPERPARAMS` in the evaluation/scripts):

| Environment | `p` | `lambda` | `alpha` |
|---|---|---|---|
| Hopper-v3 | 0.25 | 0.001 | 0.0001 |
| Walker2d-v3 | 0.25 | 0.01 | 0.0001 |
| Reacher-v2 | 0.50 | 0.001 | 0.0001 |
| HalfCheetah-v3 | 0.50 | 0.01 | 0.0001 |
| SelfishMining | 0.25 | 0.001 | 0.0001 |
| CageChallenge2 | 0.50 | 0.01 | 0.0001 |
| Macro-v1 | 0.25 | 0.01 | 0.0001 |

Table 4 mask-training sample budgets (and the timing reference used for the
~16.8 % efficiency trend check):

| Environment | Mask samples | StateMask (s) | Ours (s) |
|---|---|---|---|
| Hopper-v3 | 300 000 | 15 393 | 12 426 |
| Walker2d-v3 | 300 000 | — | — |
| Reacher-v2 | 300 000 | — | — |
| HalfCheetah-v3 | 300 000 | — | — |
| SelfishMining | 1 500 000 | — | — |
| CageChallenge2 | 10 000 000 | 79 382 | 65 400 |
| Macro-v1 | 2 443 260 | — | — |

Network architectures (mask net mirrors the target agent, §C.2/addendum):
MuJoCo `(64, 64)`; SelfishMining `(128, 128, 128, 128)`; CageChallenge2
`(64, 64, 64)`; Macro-v1 `(256, 256)` (DI-engine VAC template).

---

## 6. Success criteria (trends, not numbers)

* **Experiment I** — `fidelity(ours) ≈ fidelity(StateMask) > fidelity(Random)`
  at every `K`; mask-training wall-clock ≈16.8 % lower than the StateMask
  reference on average.
* **Experiment II** — RICE gives the largest final-reward improvement on every
  dense task; PPO fine-tuning gains are marginal; StateMask-R may be *worse*
  than No Refine. On the sparse tasks, RICE beats the baselines both in final
  performance and refining efficiency.
* **Experiment III** — `ours ≳ StateMask > Random` on all dense tasks. The
  paper's claim that ours beats StateMask *everywhere* is judged insignificant
  and is **ignored** per the addendum (trend checks treat `ours ≈ statemask` as
  acceptable).
* **Experiment IV** — RICE outperforms PPO-FT / StateMask-R / JSRL / SAC-FT when
  refining a GAIL-approximated policy; SAC continued fine-tuning stays stuck at
  the bottleneck while switching to PPO breaks through.
* **Experiment V** — `p = 0` and `p = 1` are worse than `0 < p < 1`, with
  `p = 0.25` or `0.5` best; any `lambda > 0` beats `lambda = 0` and the method
  is insensitive to `lambda` (`0.01` best except selfish mining); fidelity is
  insensitive to `alpha` (judged against seed noise).
* **Secondary** — RICE > SIL on the four MuJoCo tasks (Table 5) and
  `RICE(Ours) > AIRS > Integrated Gradients > Random` (Table 6).

All of these are checked automatically (`trend_check` in
`rice/evaluation/refining_eval.py`, `hyperparam_sweep.py`, and the drivers) and
the verdicts are written into the JSON/notes artifacts. Reference values from
Table 1/4/5 are embedded as constants purely for these checks.

---

## 7. Tests

```bash
python -m pytest rice/tests -q        # or:  python main.py test
```

| Test module | Checks |
|---|---|
| `test_mask_network.py` | Eq. (1) masking rule (keep vs. random action), `importance = P(mask=0)` ∈ [0,1], bonus `R' = R + alpha*a^m` on blinded steps, Algorithm 1 sample-budget bookkeeping, **non-trivial blinding** (mean mask rate not collapsed to 0), black-box target-policy callable, `alpha = 1e-4` default |
| `test_mixed_init.py` | mixture weights `(p, 1-p)`, **one `RAND_NUM` draw per outer iteration**, realised critical fraction ≈ `p` (β ≡ p) over a Monte-Carlo run, roll-in length `K`, degenerate `p ∈ {0, 1}` warnings, reproducibility, gym/gymnasium APIs |
| `test_rnd.py` | frozen target `f` / trainable `f_hat`, bonus on `s_{t+1}`, `R' = R + lambda R^RND`, normalisation, **bonus decays as coverage grows**, state-dict round-trip |
| `test_fidelity_score.py` | closed-form `log(d/d_max) - log(l/L)`, `l = L*K` window width, best-window argmax with earliest-tie, uniform-random window baseline, end-to-end evaluator on a dummy env |
| `test_env_reset.py` | snapshot capture/restore within tolerance, deterministic next transition after restore, feature detection & graceful degradation, snapshot pool capacity/persistence, duck-typed aliases |
| `tests/_helpers.py` | shared dependency-free envs/policies so the suite runs without MuJoCo or a trained policy |

Behavioural tests are dependency-tolerant: they skip (rather than fail) when
torch, gym or an optional module is unavailable.

---

## 8. Deviations caused by unspecified details

The paper leaves the following unspecified. Defaults are chosen from
Stable-Baselines3 / Burda et al. (2018) and recorded in the code and in the run
notes; they are **explicitly not** success criteria.

| Item | Default used here |
|---|---|
| PPO `lr`, clip, `n_steps`, `n_epochs`, GAE, entropy/value coefficients | Stable-Baselines3 defaults (`3e-4`, `0.2`, `2048`, `10`, `0.95`, `0.0`, `0.5`); `nn` `(64,64)` tanh for MuJoCo |
| PPO fine-tuning / SIL / JSRL learning rate | `10×` **lower** than the pre-training lr (`--lr-factor 0.1`); StateMask-R keeps the pre-training lr |
| Roll-in trajectory length `K` (Algorithm 2) | one full pre-trained-policy episode (`env.max_episode_steps`) |
| Refining budget per environment | derived from the Table 4 mask budgets (`env_steps` heuristic) |
| RND networks / predictor lr | `(64, 64)` MLP, `64`-dim output, Adam `1e-3`, MSE, normalised observations (clip 5.0) and normalised bonus (Burda et al. 2018) |
| Mask-net PPO hyper-parameters | same SB3 defaults as the target agent |
| `d_max` (fidelity) | estimated from warm-up rollouts (override with `--d-max`) |
| Integrated Gradients / AIRS internals | 32 / 16 Riemann steps, zero baseline, `sum_abs` aggregation, min-max normalised importance |
| SIL internals | buffer `2e5`, loss coefficient `1.0`, advantage-thresholded past-good experiences, 1 extra grad step |
| SAC / GAIL internals | SB3-style SAC defaults; GAIL discriminator `(256,256)`, lr `3e-4` |
| CAGE-2 host table / red success probabilities / observation encoding | documented in-module defaults (`DEFAULT_HOST_TABLE` covers every B-line exploit) |
| MetaDrive Macro-v1 simulator & reward shaping | in-repo Numpy BEV simulator with documented, non-paper constants |
| SelfishMining `alpha`, `gamma`, penalties, observation encoding | documented defaults in `SelfishMiningConfig` |
| Alpha conflict | §C.3 text says `alpha = 0.01` while **Table 3 lists `0.0001`**. Per the addendum **Table 3 is operative**: the code default is `1e-4`. Experiment V still sweeps `{0.01, 0.001, 0.0001}` to reproduce the insensitivity trend. |

Hardware: the paper used "a server with 8 NVIDIA A100 GPUs" (§C.1). The
reproduction runs on 1–2 GPUs; the MuJoCo dense/sparse tasks are light, while
CAGE-2 (1e7 mask samples) and MetaDrive (2.44e6) dominate cost. Everything can be
smoke-tested CPU-only (`--smoke`, `RICE_ALLOW_MUJOCO_FALLBACK=1`).

---

## 9. Out of scope (deliberately not implemented)

* §3.4 theory / Appendix B proofs — no code.
* **SparseWalker2d** refining and the sparse hyper-parameter sweeps (the env is
  registered for completeness but `in_scope = False`).
* **All Malware Mutation** experiments (Table 7, Appendix D) and its gym env.
* Quantitative/qualitative autonomous-driving analysis beyond the refining
  numbers (Figure 14).
* The MalConv / Tianshou stack.
* Exact numerical reproduction of the paper's values (trends only).

---

## 10. Citation

```
RICE: A Refining scheme for ReInforcement learning with Explanation.
Proceedings of the 41st International Conference on Machine Learning, PMLR 235, 2024.
Reference code: https://github.com/chengzelei/RICE
```
