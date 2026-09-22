# Zero-Shot Reinforcement Learning via Functional Reward Encodings (FRE)

Reproduction of **"Zero-Shot Reinforcement Learning via Functional Reward Encodings"**
(ICML 2024, PMLR 235) — reference implementation: <https://github.com/kvfrans/fre>.

FRE learns a *permutation-invariant transformer variational auto-encoder* that maps a small
set of `(state, reward)` samples drawn from an arbitrary reward function `eta(.)` into a
128-dimensional latent task vector `z`. A generalist `z`-conditioned policy/value function is
then trained with an off-the-shelf offline RL algorithm (**IQL**) over a prior mixture of
random unsupervised reward functions (goal-reaching / linear / MLP). At test time a brand-new
task is solved **zero-shot** by encoding a handful of reward-annotated states into `z` and
executing the conditioned policy — no further training.

---

## 1. Method overview

| Concept | Notation | Implementation |
|---|---|---|
| Reward-function prior | `p(eta)` | `fre/fre/prior.py` — mixture of goal-reaching, linear, MLP rewards |
| Encoder | `p_theta(z \| L^e)` | `fre/fre/encoder.py` — permutation-invariant transformer → Gaussian `z` (128-dim) |
| Decoder | `q_theta(eta(s) \| s, z)` | `fre/fre/decoder.py` — MLP `[512, 512, 512]` on raw-state ⊕ `z` |
| Training objective (Eq. 6) | `E[log q_theta] − beta·KL(p_theta ‖ u(z))` | `fre/fre/fre_model.py` — MSE + `beta·KL`, `beta = 0.01` |
| Offline RL | `Q(s,a,z)`, `V(s,z)`, `pi(a|s,z)` | `fre/rl/iql.py` + `fre/rl/networks.py` — z-conditioned IQL |
| Strided schedule (Alg. 1) | phase 1 → freeze → phase 2 | `fre/fre/trainer.py` |

Key idea: a *functional* reward encoding. Since the encoder consumes reward **values** at
states rather than a fixed task ID, one latent `z` can represent any reward function, with no
linearity constraint and no task-specific retraining.

### Architecture (paper §4.1–4.3, Appendix A, addendum)

- **Encoder input**: an unordered set of `K = 32` tokens, each the concatenation of a learned
  64-d state embedding and a learned 64-d reward embedding → 128-d tokens.
- **Reward discretization**: rescale the scalar reward to `[0,1]`, multiply by 32, floor →
  32 bins → learned embedding table (`fre/fre/reward_embedding.py`).
- **Transformer**: width 128, 4 attention heads, 4 blocks, MLP `128 → 256 → 128`.
  **No positional encodings, no causal mask** → permutation invariance over the set.
- **Posterior**: average pooling over the final layer → two linear heads (mean, log-std) of a
  128-d Gaussian `p_theta(z | L^e)`.
- **Decoder**: `[512, 512, 512]` MLP over `concat(raw_state, z)`, predicting rewards for
  `K' = 8` decoder states (disjoint from the encoder states).
- **RL networks**: `[512, 512, 512]` MLPs with `z` concatenated to the observation.

### Hyperparameters (Table 3 / Appendix A)

| Hyperparameter | Value |
|---|---|
| Encoder train steps | 150,000 (1,000,000 ExORL / Kitchen) |
| Policy train steps | 850,000 (1,000,000 ExORL / Kitchen) |
| `K` encode pairs | 32 |
| `K'` decode pairs | 8 |
| Reward binning | 32 bins, embedding dims 64 (reward) + 64 (state) = 128 |
| `z` dim | 128 |
| Encoder MLP / heads | `128 → 256 → 128`, 4 heads, 4 blocks |
| RL / decoder layers | `[512, 512, 512]` |
| KL weight `beta` | 0.01 |
| Discount `gamma` | 0.88 |
| Target update rate (Polyak) | 0.001 |
| AWR temperature | 3.0 |
| IQL expectile | 0.8 |
| Prior ratios | 0.33 goal / 0.33 linear / 0.33 MLP |
| HER goal sampling | current 0.2 / future 0.5 / random 0.3 |
| Optimizer | Adam, lr `1e-4`, batch size 512 |

### Evaluation protocol

- **5 random seeds × 20 episodes**, mean and std **across seeds**, returns normalized to `[0, 100]`.
- Zero-shot eval uses only **32** reward-annotated samples (FB / SF use **5120**).
- AntMaze: max 2000 steps, ant starts at the maze center. ExORL / Kitchen: max 1000 steps.

---

## 2. Repository layout

```
fre/
├── main.py                        # Entry point: train encoder, train policy, evaluate
├── config/
│   ├── default.py                 # All hyperparameters (Table 3) + prior-mixture settings
│   └── envs.py                    # Per-domain eval configs (AntMaze/ExORL/Kitchen) + Table 1 refs
├── fre/
│   ├── encoder.py                 # Permutation-invariant transformer encoder -> Gaussian z
│   ├── decoder.py                 # MLP decoder q_theta(eta(s)|s,z)
│   ├── reward_embedding.py        # Reward discretization (32 bins) + learned embedding table
│   ├── fre_model.py               # Joint encoder+decoder, Eq. 6 loss (MSE + beta*KL)
│   ├── prior.py                   # Reward prior sampler: goal-reaching (HER), linear, MLP
│   └── trainer.py                 # Strided training controller (encoder phase -> frozen policy phase)
├── rl/
│   ├── iql.py                     # IQL with z-conditioned Q(s,a,z), V(s,z), pi(a|s,z)
│   ├── networks.py                # [512,512,512] MLPs; z concatenated to observation
│   └── replay_buffer.py           # Offline dataset loading + sampling
├── baselines/
│   ├── gc_iql.py                  # GC-IQL baseline (goal-concat IQL)
│   ├── gc_bc.py                   # GC-BC baseline (MLE, log-std clamp -5.0)
│   ├── opal.py                    # OPAL re-impl (same transformer encoder) + privileged eval
│   ├── forward_backward.py        # Wrapper around controllable_agent (FB)
│   └── successor_features.py      # Wrapper around controllable_agent (SF, ICM features)
├── envs/
│   ├── antmaze_eval.py            # goal-reaching, directional, random-simplex, path tasks
│   ├── exorl_eval.py              # velocity + goal-reaching tasks; physics augmentation
│   ├── kitchen_eval.py            # 7 D4RL Kitchen subtasks
│   └── d4rl_loader.py             # Loads antmaze-large-diverse-v2, ExORL RND, Kitchen
├── scripts/
│   ├── train_fre.sh               # Launch encoder + policy training per domain
│   ├── eval_table1.sh             # Reproduce Table 1 (main comparison)
│   ├── train_priors_ablation.sh   # Table 4: FRE-{all,goals,lin,mlp,lin-mlp,goal-mlp,goal-lin}
│   └── run_baselines.sh           # Train/eval FB, SF, GC-IQL, GC-BC, OPAL-10
├── utils/
│   ├── normalization.py           # Normalized return 0-100; std over seeds
│   └── logging.py                 # MetricLogger / TableLogger / artifacts
└── README.md
```

---

## 3. Environment setup

Python **3.9+** is recommended.

```bash
cd fre
pip install -r requirements.txt
```

The manifest pins the stack used by the reproduction (see `requirements.txt` for notes):

- Core: `torch>=2.0,<2.4`, `numpy>=1.23,<2.0`
- Offline RL / envs: `gym==0.21.*`, `d4rl`, `mujoco-py`, `mujoco==2.3.*`, `dm-control`, `h5py`
- Prior noise: `opensimplex` (ant-random-simplex 2D noise tasks)
- Logging: `tensorboard`, `tqdm`

**D4RL** must be installed from a commit **predating June 2024** (the PyPI release changed
after publication):

```bash
pip install git+https://github.com/Farama-Foundation/d4rl@<commit-before-2024-06>
```

**Baselines FB / SF** are provided by `facebookresearch/controllable_agent`:

```bash
git clone https://github.com/facebookresearch/controllable_agent third_party/controllable_agent
export CONTROLLABLE_AGENT_DIR=$PWD/third_party/controllable_agent
```

**Datasets.** The loader (`fre/envs/d4rl_loader.py`) resolves datasets in this order:
`--dataset-path` → `$FRE_DATASET_DIR` (default `datasets/`) → the D4RL / ExORL APIs →
synthetic fallback (only when explicitly allowed). Point it at your data with:

```bash
export FRE_DATASET_DIR=/path/to/datasets
```

Expected datasets:

| Domain | Dataset |
|---|---|
| AntMaze | `antmaze-large-diverse-v2` (D4RL) |
| ExORL Walker | `walker` RND dataset (ExORL) |
| ExORL Cheetah | `cheetah` RND dataset (ExORL) |
| Kitchen | `kitchen-complete-v0` (D4RL) |

A single GPU is sufficient (Savio cluster used in the paper); CPU-only runs are possible but
slow given 150k/850k (1M/1M for ExORL/Kitchen) steps.

---

## 4. Quick start

```bash
# Sanity check the resolved plan without heavy imports
python fre/main.py --dry-run

# Train FRE (encoder phase -> freeze -> IQL policy phase) on one domain, one seed
python fre/main.py --agent fre --domain antmaze --stage all --seed 0 --run-dir runs/fre/antmaze/seed0 --device cuda

# Train + zero-shot evaluate
python fre/main.py --agent fre --domain antmaze --stage all --seed 0 \
    --run-dir runs/fre/antmaze/seed0 --device cuda --num-eval-episodes 20
```

Domains: `antmaze`, `exorl_walker`, `exorl_cheetah`, `kitchen`
(aliases `ant`, `walker`, `cheetah`, and group aliases `exorl`, `all` are also accepted).

Stages: `encoder` (phase 1 only) · `policy` (phase 2 only) · `train` · `eval` · `all` · `dry-run`.

`--num-eval-samples` controls the number of reward-annotated samples encoded into `z`
(**32** for FRE/GC agents — the paper's zero-shot budget; **5120** for FB/SF).

---

## 5. Reproducing the paper

### 5.1 Table 1 — main comparison

Trains and evaluates FRE plus the baselines (GC-IQL, GC-BC, OPAL-10, FB, SF) over all domains
and 5 seeds, then aggregates and prints observed values next to the published targets.

```bash
# FRE only (encoder + policy + eval), all domains, 5 seeds
bash fre/scripts/train_fre.sh

# Full Table 1 sweep across agents
bash fre/scripts/eval_table1.sh

# Baselines only (FB / SF / GC-IQL / GC-BC / OPAL-10)
bash fre/scripts/run_baselines.sh
```

Useful environment overrides for these scripts:

```bash
DOMAINS="antmaze exorl_walker exorl_cheetah kitchen" SEEDS="0 1 2 3 4" \
STAGE=all DEVICE=cuda RUN_ROOT=runs OUT_DIR=runs/table1 \
EPISODES=20 FRE_SAMPLES=32 FB_SF_SAMPLES=5120 OPAL_SKILLS=10 \
DATASET_DIR=/path/to/datasets bash fre/scripts/eval_table1.sh
```

Outputs: per-run logs `runs/table1/<agent>_<domain>_seed<seed>.log`, aggregated
`runs/table1/<agent>_table1.json`, and `report.json` inside each `--run-dir`.

**Published Table 1 reference** (normalized return, mean ± std over 5 seeds):

| Row | FRE | FB | SF | GC-IQL | GC-BC | OPAL-10 |
|---|---|---|---|---|---|---|
| antmaze-all | 52.8 ± 18.2 | 25.8 ± 19.8 | 11.8 ± 12.6 | — | — | 45.6 ± 17.0 |
| exorl-all | 51.5 ± 6.3 | 43.4 ± 9.1 | 40.9 ± 1.9 | — | — | 28.2 ± 4.0 |
| kitchen | 66 ± 3 | 3 ± 6 | 1 ± 1 | 59 ± 4 | 35 ± 9 | 26 ± 16 |
| **all** | **57 ± 9** | 24 ± 12 | 18 ± 5 | — | — | 33 ± 12 |

Per-task FRE rows to match: `ant-goal-reaching` 48.8 ± 6, `ant-directional` 55.2 ± 8,
`ant-random-simplex` 21.3 ± 4, `ant-path-loop` 67.2 ± 36, `ant-path-edges` 60.0 ± 17,
`ant-path-center` 64.4 ± 38, `exorl-walker-goals` 94 ± 2, `exorl-cheetah-goals` 58 ± 8,
`exorl-walker-velocity` 34 ± 13, `exorl-cheetah-velocity` 20 ± 2.

These constants live in `fre/config/envs.py` (`TABLE1_REFERENCE`) and
`fre/utils/normalization.py` (`TABLE1_FRE_TARGETS`, `TABLE1_AGGREGATE_TARGETS`); the scripts
compare observed results against them automatically.

### 5.2 Evaluation tasks

**AntMaze** (`antmaze-large-diverse-v2`, 2000 steps, center start, XY discretized into 32 bins):

- `ant-goal-reaching` — 5 fixed goals: bottom `(28,0)`, left `(0,15)`, top `(35,24)`,
  center `(12,24)`, right `(33,16)`; reward `-1` unless within distance 2.
- `ant-directional` — 4 targets: `vel_left (-1,0)`, `vel_up (0,1)`, `vel_down (0,-1)`,
  `vel_right (1,0)`; dot-product reward.
- `ant-random-simplex` — 5 fixed `opensimplex` seeds; `-1` baseline + height bonus + velocity bonus.
- `ant-path-{loop,edges,center}` — polyline-proximity path tasks.

**ExORL** (RND dataset, 1000 steps, physics augmentation for the encoder):

- `exorl-{walker,cheetah}-goals` — 5 fixed dataset goals; reward `-1` unless Euclidean distance
  `< 0.1` on std-normalized dims.
- `exorl-walker-velocity` — thresholds `0.1, 1, 4, 8`; reward `1` if `v >= threshold` decaying
  to `0`; reversed velocity ⇒ `0`.
- `exorl-cheetah-velocity` — run threshold `10`, walk threshold `1`, plus reverse variants.
- Physics features appended to encoder observations (never used for goal distance):
  walker `[horizontal_velocity, torso_upright, torso_height]`; cheetah `[speed]`.

**Kitchen** — 7 standard D4RL Kitchen subtasks with sparse rewards
(microwave, kettle, slide cabinet, hinge cabinet, light switch, bottom burner, top burner).

### 5.3 Table 4 — prior-mixture ablation (AntMaze)

```bash
bash fre/scripts/train_priors_ablation.sh
# or individually:
for v in all goals lin mlp lin-mlp goal-mlp goal-lin; do
  python fre/main.py --agent fre --domain antmaze --stage all --prior "$v" --seed 0 \
      --run-dir runs/table4/$v/antmaze/seed0 --device cuda
done
```

| Variant | AntMaze total |
|---|---|
| FRE-all | **47.3 ± 7** |
| FRE-goals | 26.1 ± 8 |
| FRE-lin | 31.6 ± 5 |
| FRE-mlp | 25.3 ± 8 |
| FRE-lin-mlp | 32.3 ± 5 |
| FRE-goal-mlp | 33.8 ± 15 |
| FRE-goal-lin | 46.9 ± 7 |

`FRE-all` (the uniform 0.33/0.33/0.33 mixture) must yield the highest total,
showing that **prior diversity** matters. Mixture weight tables live in
`fre/fre/prior.py::ABLATION_MIXTURES` and `fre/main.py::PRIOR_MIXTURES`.

### 5.4 Baselines in detail

| Baseline | Notes |
|---|---|
| **GC-IQL** | IQL with the goal concatenated to the observation; reward `0` if `s == goal` else `-1`; goal sampling `p_random = 0.3`, `p_geometric = 0.5`, `p_current = 0.2`. |
| **GC-BC** | MLP `[512,512,512]` + ReLU + LayerNorm; Gaussian policy, log-std clamped at `-5.0`; MLE loss `-E log pi(a|s,g)`; geometric future-goal sampling only. |
| **FB / SF** | Run via `controllable_agent`; SF uses ICM features; ExORL uses the RND dataset; both perform test-time linear regression on **5120** reward samples. |
| **OPAL-10** | Reuses FRE's transformer encoder; privileged eval samples 10 skills from `N(0, I)` and keeps the best rollout. |

### 5.5 Capability comparison (Table 2)

FRE satisfies all four desired properties, in contrast to the baselines:

| Property | FRE | FB | SF | GC-RL | OPAL |
|---|---|---|---|---|---|
| Zero-shot (no test-time training) | ✓ | ✗ | ✗ | ✗ | ✗ |
| Arbitrary reward functions | ✓ | ✓ | ✓ | ✗ | ✗ |
| No linearity constraint | ✓ | ✗ | ✗ | — | ✓ |
| Learns optimal policies | ✓ | ✓ | ✗ | ✓ | ✗ |

---

## 6. Programmatic usage

```python
import torch
from fre.config.envs import make_config
from fre.fre.encoder import Encoder
from fre.fre.decoder import Decoder
from fre.fre.fre_model import FREModel
from fre.fre.prior import make_prior_sampler

cfg = make_config("antmaze")
state_dim = 29                                   # obs dim (+ physics for ExORL)
model = FREModel.from_config(cfg, state_dim).to(cfg.device)
prior = make_prior_sampler(cfg, state_dim)

batch = prior.sample_batch(batch_size=cfg.batch_size)
out = model.loss(                                 # Eq. (6): MSE + beta * KL
    batch.encoder_states, batch.encoder_rewards,
    batch.decoder_states, batch.decoder_rewards,
)
out.loss.backward()                               # reconstruction + 0.01 * KL

z = model.encode(batch.encoder_states, batch.encoder_rewards, sample=False)  # (B, 128)
pred = model.decode(batch.decoder_states, z)      #   (B, K'=8)
```

Training the full pipeline (Algorithm 1):

```python
from fre.fre.trainer import train_fre
from fre.envs.d4rl_loader import load_offline_dataset
from fre.config.envs import make_config

cfg = make_config("antmaze")           # or exorl_walker / exorl_cheetah / kitchen
dataset = load_offline_dataset(cfg)
stats = train_fre(cfg, dataset)        # phase 1 (encoder) -> freeze -> phase 2 (IQL)
```

Zero-shot evaluation:

```python
from fre.envs.antmaze_eval import evaluate_antmaze_suite, make_iql_policy_fn

z = model.encode(enc_states, enc_rewards, sample=False)   # 32 reward-annotated pairs
results = evaluate_antmaze_suite(
    lambda task: make_iql_policy_fn(agent, z),
    num_episodes=20, seed=0,
)
print(results["antmaze-all"])          # normalized return, 0-100
```

---

## 7. Reproducing results end-to-end

```bash
# 1. Environment
cd fre && pip install -r requirements.txt
export FRE_DATASET_DIR=/path/to/datasets
export CONTROLLABLE_AGENT_DIR=$PWD/third_party/controllable_agent

# 2. Sanity check
python fre/main.py --dry-run

# 3. Train FRE on every domain (5 seeds each)
STAGE=all DEVICE=cuda bash fre/scripts/train_fre.sh

# 4. Table 1 (adds GC-IQL, GC-BC, OPAL-10, FB, SF)
EVAL_ONLY=1 DEVICE=cuda bash fre/scripts/eval_table1.sh
DEVICE=cuda bash fre/scripts/run_baselines.sh

# 5. Table 4 prior ablation
DEVICE=cuda bash fre/scripts/train_priors_ablation.sh

# 6. Inspect aggregates
cat runs/table1/*_table1.json
cat runs/table4/table4_summary.json
```

Expected artifacts:

```
runs/
├── <agent>/<domain>/seed<N>/report.json   # per-seed results (rows or results)
└── table1/
    ├── <agent>_<domain>_seed<N>.log
    ├── <agent>_table1.json
    └── baselines_summary.json
```

---

## 8. Notes on ambiguities and defaults

Wherever the paper is silent, the code uses documented defaults:

- **Transformer FFN activation**: GELU (the paper does not state an activation).
- **Encoder blocks**: 4 (the appendix `Encoder Layers=[256,256,256,256]` is read as the MLP
  hidden width, so each block expands `128 → 256 → 128`).
- **Reward embedding dims**: state 64 + reward 64 = **128** total (the appendix statement
  "Reward Embedding Dim=128" refers to the concatenated token width).
- **Decoder `K'` = 8**, disjoint from the encoder's `K = 32`.
- **Policy log-std clamp**: `[-5.0, 2.0]` (only the GC-BC lower clamp `-5.0` is specified).
- Reward rescaled/clipped to `[-1, 1]` before the 32-bin discretization.
- Return normalization uses per-task theoretical bounds since the paper gives no explicit bounds;
  the `success=True` short-circuit is opt-in.
- **Qualitative Figure 3 / §5.1 transfer illustrations are out of scope** (non-judgeable).

---

## 9. Citation

```bibtex
@inproceedings{frans2024fre,
  title     = {Zero-Shot Reinforcement Learning via Functional Reward Encodings},
  author    = {Frans, Kevin and Park, Seohong and Abbeel, Pieter and Levine, Sergey},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  year      = {2024}
}
```
