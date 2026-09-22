# FRE — Zero-Shot Reinforcement Learning via Functional Reward Encodings

Reference implementation of the paper **"Zero-Shot Reinforcement Learning via Functional
Reward Encodings"** (Frans et al.).

FRE learns a *functional* reward encoder: a permutation-invariant transformer variational
information bottleneck maps a set of reward-labelled states (a "context set") of an arbitrary
reward function `eta` to a 128-d latent task vector `z`. That latent then conditions an IQL
offline-RL agent (policy `pi(a|s,z)`, critic `Q(s,a,z)`, value `V(s,z)`) which is trained over a
mixture of *random* unsupervised reward functions. At test time, ~32 reward-annotated states of a
novel task are encoded into `z` and the policy acts **zero-shot** — no further training.

Original reference release: <https://github.com/kvfrans/fre>

---

## 1. Method summary (what the code implements)

### 1.1 Encoder `p_theta(z | .)` — `fre/models/encoder.py`
* Input: `K` state/reward pairs `(s_1^e, eta(s_1^e)), ..., (s_K^e, eta(s_K^e))`, `K = 32`.
* Token construction (addendum):
  * scalar reward → rescaled to `[0, 1]` → `* 32` → `floor` → learned 32-entry embedding table
    (indices clipped to `[0, 31]`; clipping is *not* specified in the paper),
  * observation → learned linear projection,
  * state embedding (64-d) ‖ reward embedding (64-d) = **128-d token**.
* 4 pre-norm transformer blocks, 4 attention heads, MLP block expands 128 → 256 → 128.
  **No positional encodings, no causal masking** → the input is an unordered set.
* Mean pooling over the `K` tokens, then two linear heads → `mu`, `log_std` of a 128-d Gaussian.
* `z = mu + sigma * eps` during training (reparameterization); posterior mean at evaluation.

### 1.2 Decoder `q_theta(eta(s^d) | s^d, z)` — `fre/models/decoder.py`
* Feed-forward MLP `[512, 512, 512]`.
* Input is the **raw** decoding state (no embedding step) concatenated with `z`.
* Each decoding state is scored independently with the shared `z`.
* Decoding states are **disjoint** from the encoding states within an update.

### 1.3 Objective (Equation 6) — `fre/training/fre_trainer.py`

```
I(L_eta^d ; Z) - beta * I(L_eta^e ; Z)
  >= E[ sum_{k=1..K'} log q_theta(eta(s_k^d) | s_k^d, z) - beta * D_KL(p_theta(z|L_eta^e) || u(z)) ]
```

`u(z)` is the unit Gaussian. The reconstruction term is realized as **negative MSE** (the paper
trains "minimizing mean-squared error between the predicted and true rewards under the decoding
states"), and `beta = 0.01`. Implemented loss (minimized):

```
loss = MSE + beta * KL( N(mu, sigma) || N(0, I) )
```

### 1.4 Prior reward distribution `p(eta)` — `fre/reward_priors/`
A uniform mixture (0.33 / 0.33 / 0.33) of three random unsupervised function classes:

| class | file | sampling |
|---|---|---|
| singleton / goal-reaching | `goal_reaching.py` | HER goal: 0.2 current state, 0.5 future state (geometric lookahead), 0.3 random state; reward `-1` until the goal is reached, `0` on success; done mask on success; context set always contains ≥1 goal sample |
| random linear | `linear.py` | `w ~ U[-1,1]`, per-dimension 0.9 probability of being zeroed (sparsity); on AntMaze the XY position dims are removed |
| random MLP | `random_mlp.py` | `(state_dim, 32, 1)`, weights `~ N(0, 1/sqrt(mean fan))`, `tanh` between layers, output clipped to `[-1, 1]` |

Named variants (Table 4 / §5.4, `fre/reward_priors/mixture.py`): `FRE-all`, `FRE-goals`,
`FRE-lin`, `FRE-mlp`, `FRE-lin-mlp`, `FRE-goal-mlp`, `FRE-goal-lin`, and `FRE-hint` (a prior
that is a *superset* of the evaluation tasks: unit (x,y) movement directions for ant-directional,
specific velocities for cheetah/walker velocity).

### 1.5 FRE-conditioned offline RL (IQL) — `fre/models/rl_networks.py`, `fre/training/iql.py`
`z` is simply concatenated to the observation fed to the RL components. IQL (Kostrikov et al.,
2021) is used with `expectile = 0.8`, `AWR temperature = 3.0`, target update rate `tau = 0.001`,
`gamma = 0.88`, Adam `lr = 1e-4`.

### 1.6 Strided two-phase training (Algorithm 1) — `fre/training/strided.py`
1. **Phase 1** — train *only* encoder + decoder on Equation (6). RL components are untouched.
2. After the encoder loss converges, **freeze the encoder** (`requires_grad=False`, `.eval()`),
   then **Phase 2** — train `pi(a|s,z)`, `Q(s,a,z)`, `V(s,z)` with IQL on the frozen latents.

Freezing makes the `eta -> z` map stationary, which the paper reports is important for correctly
estimating multitask Q-values with TD learning.

---

## 2. Repository layout

```
fre/
  fre/
    models/            # encoder.py (VIB transformer), decoder.py (reward decoder), rl_networks.py (Q/V/pi)
    reward_priors/     # goal_reaching.py, linear.py, random_mlp.py, mixture.py (+ FRE-hint)
    training/          # fre_trainer.py (Eq. 6), iql.py (expectile/AWR), strided.py (Algorithm 1)
    data/              # replay.py, antmaze_dataset.py, exorl_dataset.py, kitchen_dataset.py
    envs/              # antmaze_tasks.py, exorl_tasks.py, kitchen_tasks.py (evaluation rewards)
    eval/              # zero_shot_eval.py, baselines/{gc_bc,gc_iql,opal}.py
    utils/             # reward_discretize.py, normalization.py, logging.py
  configs/             # fre_antmaze.yaml, fre_exorl.yaml, fre_kitchen.yaml, prior_ablations.yaml
  scripts/             # train_fre.py, train_policy.py, evaluate.py, run_prior_ablation.py
  requirements.txt
  README.md
```

---

## 3. Installation

Python 3.9–3.10 with a CUDA build of PyTorch.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Notes:
* **D4RL must be pinned to a commit predating June 2024** (the upstream repo changed after the
  paper; the addendum requires the older revision for reproducibility). It supplies
  `antmaze-large-diverse-v2` and the Kitchen datasets.
* ExORL walker/cheetah experiments need `dm_control` plus the RND exploratory datasets.
  Set `FRE_EXORL_DATASET_DIR` to the directory holding them (or pass `--dataset-path`).
* `opensimplex` is required for the ant-random-simplex evaluation fields (seeds 1–5); a
  deterministic fallback noise field is used if it is missing.
* The loaders can also read raw HDF5/NPZ dumps directly via `--dataset-path`
  (AntMaze: `FRE_ANTMAZE_DATASET`; Kitchen: `FRE_KITCHEN_DATASET`), which avoids a hard D4RL
  dependency for the data pipeline.

---

## 4. Hyper-parameters (Table 3 of the paper)

| hyper-parameter | value |
|---|---|
| Batch size | 512 |
| Encoder training steps | 150,000 (1M for ExORL/Kitchen) |
| Policy training steps | 850,000 (1M for ExORL/Kitchen) |
| Reward pairs to encode (K) | 32 |
| Reward pairs to decode (K') | 8 |
| Ratio goal-reaching / linear / MLP | 0.33 / 0.33 / 0.33 |
| Number of reward embeddings | 32 |
| Reward embedding dim | 128 (= 64 state + 64 reward, addendum correction) |
| Latent dim `z` | 128 |
| Encoder layers (MLP dims) | [256, 256, 256, 256] |
| Encoder attention heads | 4 |
| Decoder network layers | [512, 512, 512] |
| RL network layers | [512, 512, 512] |
| Optimizer / Learning rate | Adam / 0.0001 |
| `beta` KL weight | 0.01 |
| Target update rate | 0.001 |
| Discount factor | 0.88 |
| AWR temperature | 3.0 |
| IQL expectile | 0.8 |

Evaluation protocol (§5.2): **20 evaluation episodes**, **5 training seeds**, standard deviation
across seeds reported, returns normalized to **0–100**. FRE encodes **32** context pairs;
FB/SF use **5120**.

---

## 5. Running the experiments

### 5.1 Phase 1 — train the encoder/decoder (Equation 6)

```bash
python scripts/train_fre.py --domain antmaze --config configs/fre_antmaze.yaml \
    --steps 150000 --save runs/fre_antmaze_seed0.pt
python scripts/train_fre.py --domain exorl --config configs/fre_exorl.yaml \
    --steps 1000000 --use-encoder-inputs --save runs/fre_exorl_seed0.pt
python scripts/train_fre.py --domain kitchen --config configs/fre_kitchen.yaml \
    --steps 1000000 --save runs/fre_kitchen_seed0.pt
```

Useful flags: `--seed`, `--device`, `--all-seeds`, `--limit`, `--log-interval`, `--skip-gates`,
`--manual-loop`, `--dry-run` (a torch/light-weight smoke test that exercises the pipeline on mock
data), `--no-strict` (checkpoint loading).

Phase-1 correctness gates are executed automatically:
1. permutation invariance — shuffling the `K` context tokens must leave `z` unchanged,
2. encoding/decoding state sets are disjoint,
3. latent dimension is exactly 128.

### 5.2 Phase 2 — train the z-conditioned IQL policy (frozen encoder)

```bash
python scripts/train_policy.py --domain antmaze --config configs/fre_antmaze.yaml \
    --checkpoint runs/fre_antmaze_seed0.pt --steps 850000 \
    --save runs/fre_antmaze_policy_seed0.pt
```

The script asserts the encoder is frozen, verifies `z` is stationary across repeated encodings,
and enforces that ExORL physics-augmented observations are used **only** by the encoder (never by
Q/V/policy, and never in goal distance).

Or use the strided controller directly:

```python
from fre.training import train_fre_strided
trainer, result = train_fre_strided(encoder, decoder, replay_buffer, prior, domain="antmaze")
```

### 5.3 Zero-shot evaluation

```bash
# FRE
python scripts/evaluate.py --domain antmaze --method FRE \
    --checkpoint runs/fre_antmaze_seed0.pt \
    --policy runs/fre_antmaze_policy_seed0.pt --suites ant-goal-reaching,ant-directional

# Baselines
python scripts/evaluate.py --domain antmaze --method GC-IQL --checkpoint runs/gciql.pt
python scripts/evaluate.py --domain antmaze --method GC-BC  --checkpoint runs/gcbc.pt
python scripts/evaluate.py --domain exorl  --method OPAL-10 --checkpoint runs/opal.pt
```

`--no-env` runs the harness without a simulator (useful for CI/smoke tests). `--save` writes a
JSON report including protocol metadata.

### 5.4 Prior-scaling ablation (Table 4)

```bash
python scripts/run_prior_ablation.py --config configs/prior_ablations.yaml \
    --variants FRE-all,FRE-goals,FRE-lin,FRE-mlp,FRE-lin-mlp,FRE-goal-mlp,FRE-goal-lin,FRE-hint \
    --seeds 0,1,2,3,4 --save results/prior_ablations.json
```

All variants share an identical training budget (addendum). The script aggregates the per-suite
scores into the Table 4 columns and checks the success criterion (FRE-all should reach the
highest total).

---

## 6. Benchmarks and tasks

### AntMaze (`configs/fre_antmaze.yaml`)
* Offline dataset: `antmaze-large-diverse-v2`, XY coordinates discretized into **32 bins**
  (shared by FRE, GC-IQL, GC-BC, OPAL).
* Episodes: **2000** steps; the ant starts at the **maze center**.
* Goal-reaching: reward `-1` until within **distance 2** of the goal, then `0`.
  Five fixed goals: `goal-bottom (28,0)`, `goal-left (0,15)`, `goal-top (35,24)`,
  `goal-center (12,24)`, `goal-right (33,16)`.
* Directional: dot product with actual velocity for `(-1,0)`, `(0,1)`, `(0,-1)`, `(1,0)`.
* Random-simplex: opensimplex fields with seeds 1–5; baseline `-1` plus bonuses for height and
  preferred velocity.
* Path tasks: `path-center`, `path-loop`, `path-edges` corridor rewards.

### ExORL (`configs/fre_exorl.yaml`)
* RND datasets for `cheetah` (`run`, `walk`, `run-backwards`, `walk-backwards`) and `walker`
  (`run`, `walk`); episodes of **1000** steps.
* Physics augmentation appended to the **encoder input only**: walker `horizontal_velocity`,
  `torso_upright`, `torso_height`; cheetah `speed`.
* Velocity tasks: cheetah thresholds 10 and 1, walker thresholds 0.1, 1, 4, 8; the reward is 1 at
  the threshold and decays linearly to 0 below it, and is 0 when moving opposite to the target.
* Goal tasks: 5 fixed dataset states, Euclidean distance **< 0.1** after per-dimension
  standardization by the offline dataset std (goal distance ignores augmented information).

### Kitchen (`configs/fre_kitchen.yaml`)
* D4RL Kitchen with the seven standard sparse subtasks
  (microwave, kettle, light switch, slide cabinet, bottom burner, top burner, hinge cabinet);
  the environment's native sparse rewards are used directly.

---

## 7. Baselines

| baseline | file | notes |
|---|---|---|
| GC-IQL | `fre/eval/baselines/gc_iql.py` | goal concatenated to the observation; rewards 0/-1; HER = 0.3 random goal / 0.5 geometric future goal / 0.2 current goal |
| GC-BC | `fre/eval/baselines/gc_bc.py` | 3×512 MLP, ReLU, LayerNorm before each activation, Gaussian head with log-std clamped at −5.0, MLE loss, geometric-only future goals |
| OPAL-10 | `fre/eval/baselines/opal.py` | no manual rewards; same transformer architecture as FRE; privileged evaluation with 10 unit-Gaussian skills and best-rollout selection |
| FB / SF | out of scope | trained/evaluated with <https://github.com/facebookresearch/controllable_agent> (DDPG policies, ICM features for SF, RND datasets for ExORL); a thin adapter hook lives in `fre/eval/baselines/__init__.py` |

---

## 8. Reference results (paper)

AntMaze (mean ± std over 5 seeds, normalized 0–100):

| method | ant-goal-reaching | ant-directional | ant-random-simplex | ant-path-loop | ant-path-edges | ant-path-center | antmaze-all |
|---|---|---|---|---|---|---|---|
| **FRE** | 48.8 ± 6 | 55.2 ± 8 | 21.3 ± 4 | 67.2 ± 36 | 60.0 ± 17 | 64.4 ± 38 | **52.8 ± 18.2** |
| FB | – | – | – | – | – | – | 25.8 ± 19.8 |
| SF | – | – | – | – | – | – | 11.8 ± 12.6 |
| OPAL-10 | – | – | – | – | – | – | 45.6 ± 17.0 |
| GC-IQL | 40.0 ± 14 | – | – | – | – | – | – |
| GC-BC | 12.0 ± 18 | – | – | – | – | – | – |

ExORL / Kitchen:

| method | exorl-walker-goals | exorl-cheetah-goals | exorl-walker-velocity | exorl-cheetah-velocity | exorl-all | kitchen |
|---|---|---|---|---|---|---|
| **FRE** | 94 ± 2 | 58 ± 8 | 34 ± 13 | 20 ± 2 | 51.5 ± 6.3 | 66 ± 3 |
| FB | – | – | – | – | 43.4 ± 9.1 | 3 ± 6 |
| SF | – | – | – | – | 40.9 ± 1.9 | 1 ± 1 |
| OPAL-10 | – | – | – | – | 28.2 ± 4.0 | 26 ± 16 |
| GC-IQL | 92 ± 4 | 100 ± 0 | – | – | – | 59 ± 4 |
| GC-BC | – | – | – | – | – | 35 ± 9 |

Aggregate "all" row: FRE 57 ± 9, FB 24 ± 12, SF 18 ± 5, OPAL-10 33 ± 12.

Table 4 (AntMaze prior scaling, equal budget): FRE-all total 47.3 ± 7 (best); FRE-goals 26.1 ± 8;
FRE-lin 31.6 ± 5; FRE-mlp 25.3 ± 8; FRE-lin-mlp 32.3 ± 5; FRE-goal-mlp 33.8 ± 15;
FRE-goal-lin 46.9 ± 7.

---

## 9. Defaults where the paper is silent

These are documented inline in the source and listed here for transparency:

* Reward-bin index clipped to `[0, 31]` (the paper's recipe can produce `32`).
* 4 transformer blocks (the appendix lists 4 encoder MLP dims).
* ReLU activations, pre-norm transformer blocks, no dropout, constant learning rate, no warmup or
  gradient clipping in phase 1.
* A single shared Adam optimizer over encoder + decoder in phase 1.
* Posterior-mean `z` and deterministic actions at evaluation (sampling during training).
* Encoder-convergence criterion: moving-average relative improvement of the Equation (6) loss
  with a minimum step floor; the Table 3 step budget is the hard cap.
* Kitchen evaluation episode length 1000; AntMaze center start is the dataset state closest to
  `(16, 16)`; AntMaze XY extent `[0, 36]`; KL/advantage clamps for numerical stability.

---

## 10. Reproducing the paper's figures/tables

1. Train the encoder/decoder for each domain and seed (`scripts/train_fre.py`) — Table 3 budgets.
2. Train the frozen-encoder IQL policy (`scripts/train_policy.py`).
3. Evaluate zero-shot over all task suites (`scripts/evaluate.py`) with 20 episodes per task and
   aggregate across 5 seeds → Table 1 / Table 3-style rows.
4. Run `scripts/run_prior_ablation.py` for Table 4 and the FRE-hint study (§5.4).
5. Table 2 (capability comparison: zero-shot, arbitrary reward functions, no linear constraint,
   learns optimal policies) is a qualitative summary of the above runs; GC-BC/GC-IQL are given the
   ground-truth goal at evaluation.

The qualitative results of Figure 3 / §5.1 are out of scope for this codebase.

Total compute implied by Table 3: 150k encoder + 850k policy steps (AntMaze), 1M + 1M steps
(ExORL/Kitchen), five seeds per agent. A single GPU with ≥16 GB is sufficient.
