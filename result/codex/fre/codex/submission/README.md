# Reproducing *Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings*

This repository is a from-scratch re-implementation of **FRE** (Frans, Park,
Abbeel, Levine; ICML 2024) and of the offline zero-shot RL baselines it is
compared against.  It contains everything needed to reproduce the paper's core
contributions: the functional reward encoding architecture, the random reward
priors, the unsupervised pre-training algorithm, and the downstream evaluation
protocol on AntMaze, ExORL and Kitchen.

No code from the original project (`kvfrans/fre`) was used.

## What the paper does

The zero-shot RL problem has two phases:

1. **Unsupervised pre-training.** Given a single unlabelled offline dataset of
   trajectories, learn a latent-conditioned policy `pi(a | s, z)` that captures
   a diverse set of behaviours, without any reward labels.
2. **Zero-shot evaluation.** Given a new downstream task defined by only a
   handful of `(state, reward)` samples, find the latent `z` that solves it and
   act, with no further training.

FRE's key idea is to learn a *functional* representation of reward functions.
The encoder is a variational transformer that maps a set of `(state, reward)`
samples produced by an arbitrary reward function `eta` to a latent `z`; the
decoder predicts `eta(s)` from `(s, z)`.  The objective is the information
bottleneck `I(L_eta^d ; Z) - beta * I(L_eta^e ; Z)`, trained over a *prior
distribution of random reward functions* (goal-reaching, random linear, and
random MLP rewards).  A generalist policy is then trained with IQL conditioned
on `z`.

## Repository layout

```
fre/
  reward_functions.py   prior distributions over random rewards (§4.2, App. B)
  encoder.py            permutation-invariant transformer VAE (§4.1)
  decoder.py            reward decoder MLP (§4.1)
  fre.py                FRE module + information-bottleneck loss (Eq. 6)
  iql.py                IQL with Q/V/policy conditioned on z (§4.1, App. A)
  training.py           strided encoder -> policy training loop (Algorithm 1)
  datasets.py           offline datasets, normalisation, HER goal sampling
  configs.py            hyperparameters (Appendix A)
  experiment.py         per-domain glue (prior + tasks + envs)
  evaluate.py           zero-shot evaluation driver (§5.2, Table 1)
  envs.py               rollout wrappers for AntMaze / ExORL / Kitchen
  utils/simplex.py      dependency-free 2D simplex noise
  tasks/
    antmaze.py          goal-reaching / directional / random-simplex / path tasks
    exorl.py            velocity + goal tasks and physics augmentation (App. C.2)
    kitchen.py          the 7 sparse Kitchen subtasks (App. C.3)
    base.py             task interfaces and return normalisation
baselines/
  gc_iql.py             Goal-Conditioned IQL
  gc_bc.py              Goal-Conditioned Behavioural Cloning
  opal.py               OPAL-style skill discovery + privileged evaluation
  iql_core.py           shared IQL update used by the goal-conditioned baseline
train_fre.py            CLI: unsupervised pre-training
evaluate_fre.py         CLI: zero-shot evaluation
train_baseline.py       CLI: GC-IQL / GC-BC / OPAL
configs/                ready-made configurations for every reported variant
scripts/                shell drivers for the full experiment matrix
docs/                   FB / SF reproduction notes
tests/                  smoke tests exercising the whole pipeline
```

## Mapping to the paper's contributions

| Paper element | Where it is implemented |
| --- | --- |
| Functional reward encoding: set-of-`(s, eta(s))` transformer VAE (§4.1) | `fre/encoder.py`, `fre/decoder.py`, `fre/fre.py` |
| Information-bottleneck objective, Eq. 6, with `beta = 0.01` | `fre/fre.py::FRE.loss`, `fre/encoder.py::gaussian_kl` |
| Reward discretisation into 32 bins + learned reward embeddings | `fre/reward_functions.py::discretize_reward`, `fre/encoder.py` |
| Prior reward distribution: goal-reaching / linear / MLP (§4.2, App. B) | `fre/reward_functions.py` |
| HER goal sampling (0.2 / 0.5 / 0.3) | `fre/datasets.py::OfflineDataset.sample_goals` |
| FRE-conditioned IQL policy, discount 0.88, expectile 0.8, AWR temp. 3.0 | `fre/iql.py`, `fre/training.py` |
| Strided training (freeze encoder before policy learning), Algorithm 1 | `fre/training.py::FRETrainer` |
| Section 5.2 / Table 1 evaluation protocol (32 samples, 20 episodes, 5 seeds) | `fre/evaluate.py`, `fre/configs.py::EvalConfig` |
| Section 5.3 scaling study over subsets of reward families (Table 4, Fig. 5) | `PRIOR_REGISTRY` in `fre/reward_functions.py`, `configs/*.json` |
| Section 5.4 domain-knowledge priors, `FRE-hint` (Fig. 6) | `fre/tasks/exorl.py::*HintPrior`, `configs/*_fre_hint.json` |
| AntMaze tasks incl. the 5 goal locations, 4 directions, 5 simplex seeds | `fre/tasks/antmaze.py` |
| ExORL velocity / goal tasks and physics augmentation | `fre/tasks/exorl.py` |
| Kitchen's seven sparse subtasks | `fre/tasks/kitchen.py` |
| GC-IQL, GC-BC, OPAL baselines (Table 1) | `baselines/` |
| FB / SF baselines (via `facebookresearch/controllable_agent`) | `scripts/run_fb_sf.sh`, `docs/fb_sf_reproduction.md` |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Environment-specific extras:

* **AntMaze / Kitchen** need `gym` + `d4rl`.  The addendum recommends a D4RL
  revision from *before June 2024* for reproducibility:
  `pip install git+https://github.com/Farama-Foundation/d4rl@<pre-June-2024-commit>`.
* **ExORL** needs `dm_control`.  Download the RND datasets with the ExORL
  downloader and point `EXORL_DATA_DIR` at them:

  ```bash
  ./download.sh walker rnd     # from github.com/denisyarats/exorl
  ./download.sh cheetah rnd
  export EXORL_DATA_DIR=$PWD/datasets
  ```

## Reproducing the experiments

### 1. Unsupervised pre-training

```bash
# AntMaze: 150k encoder steps + 850k policy steps (Appendix A)
python train_fre.py --domain antmaze --prior FRE-all --seed 0

# ExORL: 1M + 1M steps (applied automatically for walker / cheetah / kitchen)
python train_fre.py --domain walker --prior FRE-all --seed 0

# Or drive it from a config file
python train_fre.py --config configs/antmaze_fre_all.json --seed 0
```

The full matrices are in `scripts/`:

```bash
bash scripts/run_fre_antmaze.sh   # all 7 prior subsets + FRE-hint, 5 seeds
bash scripts/run_fre_exorl.sh     # walker + cheetah, FRE-all and FRE-hint
bash scripts/run_fre_kitchen.sh
```

### 2. Baselines

```bash
bash scripts/run_baselines.sh     # GC-IQL, GC-BC, OPAL (+ FB/SF instructions)
```

### 3. Zero-shot evaluation

```bash
python evaluate_fre.py \
  --domain antmaze \
  --checkpoint runs/antmaze-FRE-all-s0/policy.pt \
  --output eval/antmaze-FRE-all-s0.json

bash scripts/evaluate_all.sh      # evaluate everything and aggregate
```

`scripts/aggregate_results.py` folds the per-seed JSON files into Table 1
(mean +/- standard deviation over 5 seeds, each averaged over 20 episodes),
including the combined `antmaze-all` / `exorl-all` / `all` rows, and with
`--figure5` reproduces Figure 5's max-normalised AntMaze table (one column per
task set is 1.00 by construction, as the addendum describes).

## Implementation notes and design decisions

The paper and its addendum pin down almost every architectural detail; where a
choice was still ambiguous, the decision is documented here and in the relevant
source file.

**Encoder.** The state is projected to 64 dimensions and the reward is
discretised into 32 bins and embedded into 64 dimensions; the two are
concatenated to a 128-dimensional token, matching the addendum's correction of
the appendix's "Reward Embedding Dim 128" typo.  Four pre-LN transformer blocks
with 4 heads, a 128-dimensional residual stream and a 256-wide MLP operate over
the *unordered* set of tokens (no positional encodings, no causal mask), so the
encoder is permutation invariant.  The mean of the final representations is
mapped to the mean and log-standard-deviation of `z` (128 dimensions).

**Reward discretisation.** Rewards are rescaled to `[0, 1]` and multiplied by
32 (floor, clipped to 31 bins).  Each reward family has a known analytic range
(`[-1, 0]` for goal-reaching, `[-1, 1]` for linear / MLP, `[0, 1]` for the ExORL
velocity tasks), and the same range is used at evaluation time so that the bin
indices line up with the embedding table learned during pre-training.

**Model states vs. reward states.** FRE assumes reward functions are pure
functions of the environment state.  The models are fed *preprocessed* states,
while reward functions are evaluated on the raw state.  Two places matter:
(a) AntMaze discretises `(x, y)` into 32 bins for all four agents (§C.1);
(b) ExORL appends physics features (`horizontal_velocity`, `torso_upright`,
`torso_height` for walker; `speed` for cheetah) to the observation *for encoder
training only* (Appendix C.2), while the Q/V/policy networks use the underlying
observation space.

**Goal-reaching prior.** Goals are drawn with the HER distribution from
Appendix B (0.2 current state / 0.5 future state / 0.3 random dataset state) and
the goal state is forced into every encoding set.  The reward is `-1` until the
goal is reached and `0` afterwards.  On AntMaze the distance is measured on the
maze `(x, y)` with the same threshold (2) as the evaluation tasks; on ExORL it
is the Euclidean distance in the standard-deviation-normalised observation
space with threshold 0.1, matching the evaluation protocol.

**Random linear prior.** Uniform weights in `[-1, 1]` with an independent
Bernoulli(0.9) mask zeroing each dimension; the `(x, y)` position dimensions are
excluded on AntMaze (Appendix B).

**Random MLP prior.** Hidden size 32, weights drawn from a normal distribution
scaled by the average layer dimension, `tanh` nonlinearity, output clipped to
`[-1, 1]`.

**IQL.** The value function regresses the *target* Q with an expectile loss
(`0.8`), the Q-function uses the Bellman backup `r + gamma (1 - done) V(s')` with
`gamma = 0.88`, and the policy is extracted by advantage-weighted regression
with temperature `3.0`.  `z` is concatenated to the observation fed to all three
networks.  Following Algorithm 1, the encoder is frozen before policy training
so that the `eta -> z` mapping is stationary during TD learning.

**Task definitions.** Every task follows the addendum: the five AntMaze goal
locations `(28,0)`, `(0,15)`, `(35,24)`, `(12,24)`, `(33,16)`; the four unit
directional velocities; the five seeded simplex-noise tasks (baseline `-1`, a
height bonus, and a bonus for moving along the noise field's preferred
direction); the ExORL velocity thresholds (cheetah 10 / 1 forward and backward,
walker 0.1 / 1 / 4 / 8) with linear decay to zero below threshold and zero
reward for backwards motion; and ExORL goal-reaching at a normalised distance
threshold of 0.1 with five fixed dataset goals.  Episode limits are 2000 steps
on AntMaze and 1000 on ExORL.

**Return normalisation.** Table 1 reports scores "normalized between 0 and 100".
Each task normalises its raw return using the analytic worst and best episode
returns of its reward function (`fre/tasks/base.py::EvalTask.normalize`).
For goal-reaching tasks the best case is reaching the goal on the first step.

**Underspecified pieces (documented approximations).**

* The three `ant-path-*` corridors are described only in prose, so
  `CORRIDOR_WAYPOINTS` in `fre/tasks/antmaze.py` hand-crafts polylines from the
  `HARDEST_MAZE_TEST` free-space layout (the central corridor is the fully free
  column at `x = 20`; the edge task is the union of the four border corridors).
* The exact "centre of the maze" reset is implemented as the free maze cell
  nearest the geometric centre of the free space (`center_reset_xy`).
* The directional / simplex rewards are clipped and scaled by a reference speed
  so that they lie in a bounded range suitable for the return normalisation.
* OPAL's trajectory VAE reuses FRE's transformer blocks over `(state, action)`
  tokens, as the addendum states that OPAL uses the same encoder architecture.

## Validation performed here

`tests/test_fre_smoke.py` (run with `python -m tests.test_fre_smoke`) exercises:
dataset/transition/goal sampling, every prior family and the reward
discretisation, the encoder-decoder forward pass and the information-bottleneck
loss, a few steps of the full strided training loop, all AntMaze and ExORL task
reward functions, the GC-IQL / GC-BC / OPAL baselines, and the evaluation
driver (encoding a task, rolling out, and normalising the return) against a
stub environment.  All tests pass on CPU.

`scripts/smoke_train.py` runs a self-contained end-to-end demonstration on a
synthetic dataset (about three minutes on one CPU core).  Measured output at
800 pre-training steps plus a 300-step capacity check:

```
in-context reconstruction on the pre-training distribution (last 100 steps):
    recon=0.105 vs per-function-mean baseline=0.091
capacity check (fit one fixed reward function):
    residual mse=0.0014 vs reward variance=0.0804  -> PASS
```

The capacity check shows that the `context -> z -> decoder` path really does
carry a reward function (98% of that function's variance is explained); the
in-context numbers show that the *generalisation* of the encoder to unseen
reward functions is far from converged after a few hundred steps, which is
expected given that the paper trains for 150k-1M encoder steps (~200-1000x
more).  The demo is a plumbing and stability check, not a convergence check.

Full-scale training (150k-1M encoder steps, then 850k-1M policy steps, five
seeds per configuration, plus environment rollouts) is intended to run outside
this sandbox; no GPU is required but the compute is substantial.  The commands
are the ones in `scripts/`.

## How long a full reproduction takes

Rough per-configuration budget for the paper's numbers, assuming the training
loop in `fre/training.py` (one encoder step and one policy step each sample 512
reward functions evaluated on 32 / 8 states plus a 512-transition RL batch):

| Domain | Encoder steps | Policy steps | Notes |
| --- | --- | --- | --- |
| AntMaze (`antmaze-large-diverse-v2`, 1M transitions) | 150,000 | 850,000 | 7 prior variants + `FRE-hint` |
| ExORL walker / cheetah (RND, 5M transitions each) | 1,000,000 | 1,000,000 | `FRE-all` and `FRE-hint` |
| Kitchen (`kitchen-complete-v0`) | 1,000,000 | 1,000,000 | `FRE-all` |

Every entry is trained with 5 seeds and evaluated over 20 episodes per task,
which is what `scripts/run_fre_*.sh` and `scripts/evaluate_all.sh` do.

## What is not reproduced

* **Figure 3 of §5.1** is explicitly out of scope (the addendum excludes it as a
  qualitative discussion).
* **FB / SF** are not re-implemented, because the addendum requires those two
  columns to be produced with the upstream `controllable_agent` codebase; the
  repository instead documents and scripts exactly that workflow.
