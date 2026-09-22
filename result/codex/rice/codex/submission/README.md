# RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation

Reproduction of *Cheng, Wu, Yu, Yang, Wang, Xing — "RICE: Breaking Through the
Training Bottlenecks of Reinforcement Learning with Explanation", ICML 2024.*

RICE refines a pre-trained (bottlenecked) DRL agent with two ingredients:

1. **a step-level explanation method** — a *mask network* trained with PPO that
   blinds the agent at non-critical steps (`R'(s,a) = R(s,a) + α·a^m`, Algorithm 1);
2. **a refining algorithm** — the agent is re-trained from a *mixed initial state
   distribution* that mixes the default initial states with the critical states
   identified by the mask network, and it receives a Random Network Distillation
   exploration bonus while being optimised with PPO (Algorithm 2).

This repository implements both contributions, the three refining baselines
(PPO fine-tuning, StateMask-R, JSRL), the baseline explanation methods (Random,
StateMask), the fidelity metric, and the five experiments of Section 4.2 of the
paper.

---

## 1. Mapping between the paper and the code

| Paper item | What it is | Code |
|---|---|---|
| Alg. 1, Eq. (3)-(4), Thm. 3.3 | mask network, `R' = R + α·a^m`, importance = `P(a^m=0∣s)` | [`rice/explanation/mask_trainer.py`](rice/explanation/mask_trainer.py), [`rice/networks.py`](rice/networks.py) (`MaskNet`) |
| Sec. 3.3 "Constructing Mixed Initial State Distribution" | `μ(s) = β·d_ρ^π̂(s) + (1-β)·ρ(s)` | [`rice/refining/critical_state_provider.py`](rice/refining/critical_state_provider.py) |
| Sec. 3.3 "Exploration with RND", Alg. 2 | RND intrinsic reward `λ‖f(s')-f̂(s')‖²` + PPO | [`rice/refining/rnd.py`](rice/refining/rnd.py), [`rice/refining/trainer.py`](rice/refining/trainer.py), [`rice/refining/methods.py`](rice/refining/methods.py) |
| Sec. 4.1 "Evaluation Metrics" | fidelity score `log(d/d_max) − log(l/L)` with a sliding window `l = K·L` | [`rice/fidelity.py`](rice/fidelity.py) |
| Experiment I (Fig. 5, Table 4) | fidelity of Ours vs StateMask vs Random; mask training time | [`rice/experiments/explanation.py`](rice/experiments/explanation.py), `scripts/run_experiment1_fidelity.py` |
| Experiment II (Table 1 left, Fig. 2) | PPO fine-tuning vs JSRL vs StateMask-R vs RICE | [`rice/experiments/refining.py`](rice/experiments/refining.py), `scripts/run_experiment2_refine.py` |
| Experiment III (Table 1 right) | refining with Random / StateMask / Ours explanations | `scripts/run_experiment3_explanations.py` |
| Experiment IV (Fig. 3, Fig. 6) | SAC agent → GAIL imitation → refining (+ SAC fine-tuning baseline) | [`rice/imitation.py`](rice/imitation.py), [`rice/experiments/sac_refine.py`](rice/experiments/sac_refine.py), `scripts/run_experiment4_sac.py` |
| Experiment V (Fig. 7, 8, 9) | sensitivity of `p`, `λ` and `α` | [`rice/experiments/hyperparams.py`](rice/experiments/hyperparams.py), `scripts/run_experiment5_hyperparams.py` |
| Sec. 4.1 "Environment Selection" | 4 dense MuJoCo games, 3 sparse MuJoCo games, 4 security applications | [`rice/envs/registry.py`](rice/envs/registry.py) |
| Appendix C.1 "environment reset function" | restore the simulator to a critical state / fast-forward | [`rice/envs/adapters.py`](rice/envs/adapters.py) |

A line-by-line mapping between the pseudo-code of the two algorithms and the
code is in [`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md).

### Environments

| Application | Implementation |
|---|---|
| Hopper / Walker2d / Reacher / HalfCheetah | `gymnasium` MuJoCo (`Hopper-v4`, …); the paper used `Hopper-v3`, `Walker2d-v3`, `Reacher-v2`, `HalfCheetah-v3` |
| SparseHopper / SparseWalker2d / SparseHalfCheetah | [`rice/envs/sparse_reward.py`](rice/envs/sparse_reward.py): `reward = x` if `x > threshold` else `0` (`0.6`, `0.6`, `5.0`) |
| Selfish Mining | [`rice/envs/selfish_mining.py`](rice/envs/selfish_mining.py): 3-action MDP (adopt / reveal / mine), transaction fees 1, whale transaction fee 10 with probability 0.01, penalty for unsuccessful actions |
| CAGE Challenge 2 | [`rice/envs/cage_challenge.py`](rice/envs/cage_challenge.py): wrapper around the challenge simulator (clone `cage-challenge/cage-challenge-2` and `pip install -e .`, then set `RICE_CAGE_PATH`; the PyPI package named `cyborg` is an unrelated library) |
| Autonomous Driving (Macro-v1) | [`rice/envs/meta_drive.py`](rice/envs/meta_drive.py): wrapper around `metadrive-simulator` |
| Malware Mutation | out of scope (see §5) |

---

## 2. Quick start

```bash
pip install -r requirements.txt
python -m unittest discover -s tests          # ~100 s, 21 tests
```

Full reproduction of one application (e.g. Hopper):

```bash
# 1. pre-train the target agent (the bottlenecked policy π)
python scripts/pretrain_agents.py --envs Hopper --steps 300000 --seeds 0 1 2

# 2. train the two mask networks (Algorithm 1 and the StateMask baseline)
python scripts/train_mask_network.py --env Hopper --method ours      --agent checkpoints/agents/Hopper_seed0.pt
python scripts/train_mask_network.py --env Hopper --method statemask --agent checkpoints/agents/Hopper_seed0.pt
# (optional) check that the mask rate is in the informative regime
python scripts/calibrate_alpha.py --env Hopper --agent checkpoints/agents/Hopper_seed0.pt

# 3. Experiment I: fidelity (Fig. 5) + mask training cost (Table 4)
python scripts/run_experiment1_fidelity.py --env Hopper --seeds 3

# 4. Experiments II + III: Table 1 (left and right), Figure 2 for sparse games
python scripts/run_experiment2_refine.py --env Hopper --seeds 0 1 2
python scripts/run_experiment2_refine.py --env SparseHopper --seeds 0 1 2

# 5. Experiment V: sensitivity of p / lambda / alpha
python scripts/run_experiment5_hyperparams.py --env Hopper --seeds 0 1 2

# 6. Experiment IV: SAC + GAIL imitation, refined with all methods
python scripts/run_experiment4_sac.py --env Hopper --seeds 0 1 2

# 7. figures
python scripts/plot_results.py --results results --out figures
```

Everything at once (see `scripts/run_all.py`):

```bash
python scripts/run_all.py --envs Hopper Walker2d Reacher HalfCheetah \
    --sparse-envs SparseHopper SparseHalfCheetah --seeds 0 1 2
```

All results are written as JSON (`results/`), checkpoints as `checkpoints/`
(both are git-ignored).

---

## 3. What is implemented

**Explanation (Algorithm 1).** `MaskActorCritic` is a 2-action PPO policy.  For
every step of a trajectory sampled from π, the mask network proposes `a^m`;
if `a^m = 1` a uniformly random action is executed instead of `a_t ~ π`.  The
PPO reward is `R(s_t,a_t) + α·a^m` (Thm. 3.3 guarantees `η(π̄) ≤ η(π)`, so
maximising `η(π̄)` is equivalent to StateMask's `min |η(π) − η(π̄)|` while the
bonus prevents the trivial "never blind" solution).  The importance score of a
state is `P(a^m = 0 ∣ s)`.  The training budget of Table 4 (number of samples)
is a first-class argument of the trainer, which also returns the wall-clock
training time and the sample count for the efficiency comparison.

**Refining (Algorithm 2).** One trainer implements all methods; only the
hooks change:

| Method | initial state | exploration | notes |
|---|---|---|---|
| RICE (ours) | critical state with probability `p`, otherwise `ρ` | RND, weight `λ` | `p`, `λ` from Appendix C.3 |
| StateMask-R | critical state with probability 1 | — | reproduces the overfitting behaviour |
| PPO fine-tuning | `ρ` | — | lowered learning rate |
| JSRL | guided roll-in with π (curriculum) | — | π_e initialised from π |

The critical state is obtained by rolling π out for `rollin_length` steps,
scoring every visited state with the frozen mask network and picking the
`argmax` importance (Algorithm 2).  The environment is then restored to that
state with the simulator snapshot mechanism of Appendix C.1
(`MujocoStatefulEnv`, `DictStatefulEnv`, CAGE/MetaDrive replay adapters).

**Fidelity (Experiment I).** For every trajectory: score the steps, take the
window of width `l = K·L` with the highest average importance, fast-forward to
its first state, execute `l` *random* actions, then let π finish the episode;
`d = |R' − R|` and `score = log(d/d_max) − log(l/L)`.  500 trajectories × 3
seeds × `K ∈ {10,20,30,40}%`, exactly as described in the paper and its
addendum.

**Practical note on `α`.**  The bonus `α·a^m` has two degenerate regimes: with
raw rewards an `α` of 0.01 is negligible next to, e.g., the ~3.6 reward per step
of Hopper and the mask never blinds (every importance score ≈ 1); with a very
large `α` it blinds everything (every score ≈ 0).  The paper's SB3 pipeline
normalises rewards, which keeps `α = 0.01` meaningful, and this repository does
the same by default (`EnvSpec.mask_reward_scale = None` → divide by the running
standard deviation of the episode returns; set it to `1.0` for the raw-reward
formula of the paper).  `docs/EXPERIMENTS.md` reports the measured mask
rate/importance spread and the fidelity of each setting; on the selfish mining
application a non-degenerate mask is obtained with normalised rewards and
`α = 0.05` (the registry's per-application value) and its fidelity beats the
random baseline at every `K`, as in the paper.

---

## 4. Expected results (reference values from the paper)

Table 1 of the paper (mean (std) final reward after refining, fixed explanation
= ours):

| Task | No Refine | PPO | JSRL | StateMask-R | **Ours** | Random expl. | StateMask expl. |
|---|---|---|---|---|---|---|---|
| Hopper | 3559.44 | 3638.75 | 3635.08 | 3652.06 | **3663.91** | 3648.98 | 3661.86 |
| Walker2d | 3768.79 | 3965.63 | 3963.57 | 3966.96 | **3982.79** | 3969.64 | 3982.67 |
| Reacher | −5.79 | −3.04 | −3.23 | −3.45 | **−2.66** | −3.11 | −2.69 |
| HalfCheetah | 2024.09 | 2133.31 | 2128.04 | 2085.28 | **2138.89** | 2132.01 | 2136.23 |
| Selfish Mining | 14.36 | 14.93 | 14.88 | 14.53 | **16.56** | 15.09 | 16.49 |
| Cage Challenge 2 | −23.64 | −23.58 | −22.97 | −26.98 | **−20.02** | −25.94 | −20.07 |
| Auto Driving | 10.30 | 13.37 | 11.26 | 7.62 | **17.03** | 11.72 | 16.28 |

Trends to check in a reproduction (the grading criterion of the paper's
addendum): (i) RICE ≥ every baseline in every application; (ii) PPO
fine-tuning only improves marginally; (iii) StateMask-R can *hurt* (Cage
Challenge 2); (iv) RICE with our explanation ≥ RICE with the StateMask
explanation ≥ RICE with a random explanation; (v) fidelity(Ours) ≈
fidelity(StateMask) > fidelity(Random); (vi) mask training with our objective
is ~16.8 % faster than StateMask for the same sample budget; (vii) `p` in
{0.25, 0.5} beats `p ∈ {0, 1}` and any `λ > 0` beats `λ = 0`; the method is
insensitive to `α`.

Reference numbers of the remaining tables are stored in
[`configs/paper_hyperparameters.json`](configs/paper_hyperparameters.json)
(Table 4 timings) and [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)
(Tables 5-6, Figure 2/3).

---

## 5. Scope, approximations and caveats

**Out of scope** (per the paper's addendum — intentionally not reproduced):
Section 3.4 (theoretical analysis), the refining results of SparseWalker2d's
hyper-parameter sensitivity, the hyper-parameter sensitivity of the sparse
MuJoCo games, the qualitative analysis of the autonomous driving case, every
experiment specific to the Malware Mutation application (Appendix D), and the
appendix-only comparisons with Self-Imitation Learning (Table 5) and
Integrated-Gradients/AIRS explanations (Table 6).  Their supporting code paths
(`SelfishMining`, `SparseWalker2d`, `AutoDriving` refining) are still present
because they share the same drivers.

**Approximations.**

* *StateMask baseline.*  We re-implement StateMask's primal-dual objective
  (`r_t^m = a_t^m − λ|G_pert − G_target|`, `λ` updated by dual ascent) instead of
  vendoring the authors' code; the extra roll-out of the target policy that it
  requires is what makes it slower than our method, matching Table 4.  A
  checkpoint trained with the official implementation can be dropped in with
  `rice.explanation.mask_io.load_mask_net`.
* *Selfish Mining.*  Bar-Zur et al.'s full blockchain simulator is replaced by a
  compact MDP with the same state variables, the same three actions and the same
  fee structure; the pre-train → explain → refine pipeline is identical.
* *Gymnasium version.*  `Hopper-v3`/`Reacher-v2` are not shipped by modern
  `gymnasium`; the registry uses `-v4` ids (same dynamics, same episode length).
* *`d_max`.*  The maximum per-episode reward of each environment is a constant
  in `rice/envs/registry.py`.  It shifts all fidelity scores by a constant, so
  the comparison between explanation methods (the actual claim of Experiment I)
  is unaffected.
* *Compute.*  The reproduction environment has no GPU, so only smoke-scale runs
  were executed here (~160 policy steps/s for MuJoCo on a single CPU core);
  the full experiments are meant to be launched on the machine described in
  the README with the step budgets of `configs/paper_hyperparameters.json`.
  `docs/SMOKE_RESULTS.md` §7-8 states precisely what was verified (the
  explanation/fidelity trend on selfish mining) and what needs the full budget
  (the dense MuJoCo applications, whose agents never reach the paper's
  bottleneck level in one CPU hour).

**Verified locally.**  `python -m unittest discover -s tests` (21 tests) passes;
the end-to-end pipeline (pre-train → mask training → RICE / JSRL / PPO /
StateMask-R refining → fidelity) was executed on `Hopper` and on
`SelfishMining` with reduced budgets, producing the expected qualitative signal
(mask-based explanation > random explanation in fidelity).

**Reproducibility.**  Every entry point takes a `--seed`; the trainers call
`rice.utils.set_global_seeds`, which seeds python, numpy, torch and the
environment (including its action space, used by the random actions of
Algorithm 1 and of the fidelity metric), so repeated runs with the same seed
give identical results.  Recorded smoke-run outputs are in
[`docs/SMOKE_RESULTS.md`](docs/SMOKE_RESULTS.md).

## 6. Repository layout

```
rice/
  networks.py            MLP actor-critic + binary mask network
  ppo_core.py            PPO buffer/updater shared by every learner
  rollout.py             episode rollouts with state snapshots
  fidelity.py            fidelity score of Experiment I
  running_stats.py       running mean/std (RND and observation normalisation)
  training.py            pre-training the target agents (PPO / SB3)
  imitation.py           GAIL imitation (Experiment IV)
  envs/                  registry, state adapters, sparse rewards, applications
  explanation/           Algorithm 1, StateMask baseline, critical states
  refining/              Algorithm 2, the three baselines, RND
  experiments/           the five experiments of the paper
scripts/                 CLI entry points (pre-train, explain, refine, plot)
tests/                   unit / smoke tests
configs/                 hyper-parameters and reference numbers
docs/                    experiment notes, runbook, algorithm mapping, smoke runs
```

The most useful documents are
[`docs/RUNBOOK.md`](docs/RUNBOOK.md) (the command matrix for a GPU machine),
[`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md) (pseudo-code ↔
code), [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) (what each script
reproduces) and [`docs/SMOKE_RESULTS.md`](docs/SMOKE_RESULTS.md) (recorded runs).
