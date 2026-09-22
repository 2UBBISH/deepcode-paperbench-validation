# Zero-Shot Reinforcement Learning via Functional Reward Encodings (FRE)

This repository reproduces **Functional Reward Encodings (FRE)** — a zero-shot
offline RL method that represents *arbitrary* reward functions as a single
128-dimensional latent task embedding `z`.

The core idea: a transformer-based variational auto-encoder (the **FRE encoder**)
maps a small *unordered set* of `K = 32` `(state, reward)` pairs
`(s_i, eta(s_i))` into a Gaussian posterior `p_theta(z | context)`. A reward
**decoder** `q_theta(eta(s) | s, z)` reconstructs the reward function from `z`, and
a `z`-conditioned IQL policy is pretrained over a *mixture of random unsupervised
reward functions*. At test time, only 32 `(state, reward)` samples of a brand-new
task are needed to encode `z` and act **zero-shot** on it.

---

## 1. Method summary

| Component | Description | Paper location |
|---|---|---|
| **FRE encoder** | Permutation-invariant transformer (no positional encoding, no causal mask) over `K=32` tokens; each token = 64-d state projection ⊕ 64-d reward-bin embedding = 128-d | Sec. 4.1, App. A |
| **FRE decoder** | MLP `[512, 512, 512]` mapping `concat(raw_state, z) -> scalar reward`, applied independently to `K' = 8` held-out states | Sec. 4.1 |
| **Info-bottleneck objective** | `MSE(eta_hat, eta) + beta * KL(q(z|ctx) || N(0,I))`, `beta = 0.01` | Eq. (6) |
| **Reward prior `p(eta)`** | Uniform mixture `0.33 goal-reaching / 0.33 masked-linear / 0.33 random-MLP` | Sec. 4.2, App. B |
| **z-conditioned IQL** | `Q(s,a,z)`, `V(s,z)`, `pi(a|s,z)`; expectile `0.8`, AWR temp `3.0`, `gamma = 0.88`, target rate `0.001`; `z` concatenated to observations | Sec. 4.3 |
| **Strided training** | Phase 1: encoder+decoder (150k AntMaze / 1M ExORL+Kitchen). Phase 2: freeze encoder, train IQL (850k AntMaze / 1M) | Algorithm 1 |
| **Zero-shot eval** | Encode exactly **32** `(state, reward)` pairs -> `z`; roll out `20` episodes x `5` seeds; report normalized return in `[0,100]` | Sec. 5 |

Baselines implemented in-repo: **GC-IQL**, **GC-BC**, **OPAL(-10)**.
FB / SF baselines are run via the external
[`facebookresearch/controllable_agent`](https://github.com/facebookresearch/controllable_agent)
repository (see §5).

---

## 2. Repository structure

```
fre/
├── main.py                  # orchestrator: encoder -> policy -> eval (+ baselines) per domain
├── train_encoder.py         # Phase 1: FRE encoder/decoder pretraining driver
├── train_policy.py          # Phase 2: z-conditioned IQL policy training driver
├── evaluate.py              # zero-shot evaluation harness (5 seeds x 20 episodes)
├── configs/
│   ├── antmaze.yaml         # AntMaze-large-diverse-v2
│   ├── exorl.yaml           # ExORL RND walker / cheetah
│   └── kitchen.yaml         # D4RL Kitchen
├── fre/
│   ├── encoder.py           # permutation-invariant transformer (Sec 4.1)
│   ├── decoder.py           # q_theta(eta(s)|s,z) MLP [512,512,512]
│   ├── vae_loss.py          # info-bottleneck objective Eq.(6): MSE + beta*KL
│   ├── reward_embeddings.py # rescale -> *32 -> floor discretization + 32x64 table
│   └── latent_policy.py     # z-conditioned IQL nets Q/V/pi (concat z to obs)
├── rewards/
│   ├── base.py              # RewardFunction / RewardFunctionPrior interfaces
│   ├── prior.py             # mixture p(eta) 0.33/0.33/0.33 (+ ablations, hints)
│   ├── goal_reaching.py     # HER singletons (-1 until goal, 0 else)
│   ├── linear.py            # uniform[-1,1] vector + 0.9 sparsity mask
│   ├── mlp.py               # (state_dim, 32, 1) tanh, output clipped [-1,1]
│   └── eval_rewards.py      # AntMaze / ExORL / Kitchen task reward functions + scoring
├── rl/
│   ├── iql.py               # IQL: expectile 0.8, AWR temp 3.0, gamma 0.88
│   └── replay_buffer.py     # offline transitions, HER trajectory indexing
├── envs/
│   ├── antmaze_wrapper.py   # XY 32-bin discretization, center start, dist <= 2 goal
│   ├── exorl_wrapper.py     # physics append, std-normalization, dist < 0.1 goal
│   └── kitchen_wrapper.py   # 7 sparse subtasks from obs[-7:] flags
├── data/
│   ├── d4rl_loader.py       # antmaze-large-diverse-v2, Kitchen
│   └── exorl_loader.py      # RND datasets (walker, cheetah)
├── baselines/
│   ├── gc_iql.py            # goal-conditioned IQL
│   ├── gc_bc.py             # goal-conditioned BC
│   └── opal.py              # OPAL-10 (privileged best-of-10 skills)
├── utils/                   # discretize.py, normalization.py, logging.py
└── requirements.txt
```

---

## 3. Installation

Python **3.8–3.10** (3.9 recommended) with a CUDA-capable GPU (>= 16 GB per seed
for the full 1M + 1M step runs).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3.1 MuJoCo (required for AntMaze / Kitchen / live ExORL)

```bash
# MuJoCo 2.1.0 binaries
mkdir -p ~/.mujoco && cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so   # if needed
```

### 3.2 D4RL — **install from a commit dated before June 2024**

> ⚠️ **Critical:** the post-June-2024 D4RL release changed datasets/environments.
> Pin an older commit for exact reproduction.

```bash
pip install git+https://github.com/Farama-Foundation/d4rl@<PRE_JUNE_2024_SHA>
# then fix the gym/mujoco bindings if necessary:
pip install gym==0.21.0
```

Datasets are downloaded automatically on first `gym.make(...).get_dataset()`
(AntMaze) / `d4rl.kitchen` (Kitchen), or placed manually under `./data`.

### 3.3 ExORL RND datasets

Download the **RND** exploratory datasets for `walker` and `cheetah` using the
official utilities and place them under `./data/` (or set `FRE_DATA_DIR`):

```bash
git clone https://github.com/denisyarats/exorl
# follow exorl/README to fetch the *rnd* datasets, then:
#   ./data/rnd_walker.npz  (or .pkl / directory)
#   ./data/rnd_cheetah.npz
```

The loader (`fre/data/exorl_loader.py`) reads local archives only; if the files
are absent the loaders raise a clear error and the wrappers fall back to
**offline dataset-replay** rollouts, which let you exercise the full pipeline
without a live simulator.

### 3.4 External FB / SF baselines (required for Table 1)

Forward-Backward (`FB`) and Successor-Features (`SF`) are **not** reimplemented
here. Run them from:

```bash
git clone https://github.com/facebookresearch/controllable_agent
# train FB / SF on the same RND data (SF uses ICM features), then map their
# returns onto the FRE normalized [0,100] scale.
```

---

## 4. Running the pipeline

The orchestrator runs, per domain: **Phase 1 (encoder) -> Phase 2 (policy) ->
Phase 3 (zero-shot eval)**, optionally followed by baselines.

```bash
# Full AntMaze run (150k encoder steps + 850k IQL steps + eval)
python -m fre.main --domain antmaze --phase all \
    --config fre/configs/antmaze.yaml

# ExORL (expands into walker + cheetah sub-domains; 1M + 1M steps each)
python -m fre.main --domain exorl --phase all --config fre/configs/exorl.yaml

# Kitchen (1M + 1M steps)
python -m fre.main --domain kitchen --phase all --config fre/configs/kitchen.yaml

# Quick smoke test (tiny number of steps / episodes)
python -m fre.main --domain antmaze --phase all --dry-run
```

Individual phases:

```bash
python -m fre.train_encoder --config fre/configs/antmaze.yaml --steps 150000
python -m fre.train_policy  --config fre/configs/antmaze.yaml \
       --encoder-checkpoint runs/antmaze/encoder.pt --steps 850000
python -m fre.evaluate      --domain antmaze \
       --checkpoint runs/antmaze/policy.pt --num-episodes 20 --num-seeds 5
```

Baselines (same data / eval protocol):

```bash
python -m fre.baselines.gc_iql --domain antmaze
python -m fre.baselines.gc_bc  --domain antmaze
python -m fre.baselines.opal   --domain antmaze --num-eval-skills 10
```

Results are written to `runs/<domain>/` plus an aggregate `fre_report.json`
containing Table-1-style scores and a comparison against the paper's numbers.

---

## 5. Key hyperparameters (all configs agree)

```
latent_dim          = 128          # z dimension
context_size (K)    = 32           # (state, reward) encoder samples
decoder_size (K')   = 8            # disjoint decoder states
token dim           = 128          # 64-d state emb + 64-d reward emb (corrected addendum)
reward bins         = 32           # rescale [0,1] -> *32 -> floor
transformer         = 4 blocks, 4 heads, MLP 128->256->128, no pos-enc, no causal mask
decoder MLP         = [512, 512, 512]
beta (KL)           = 0.01
IQL                 = gamma 0.88, expectile 0.8, AWR temp 3.0, target rate 0.001
optimizer           = Adam, lr 1e-4, batch 512 (both phases)
reward functions / RL batch = 512  (re-sampled every iteration, z re-encoded)
eval                = 20 episodes x 5 seeds, deterministic policy, 32 context samples
```

**Strided schedule (Algorithm 1).** The encoder is trained first and then
**frozen** for Phase 2, which makes the `eta -> z` mapping stationary and keeps
multitask TD learning stable.

| Domain | Encoder steps | Policy steps |
|---|---|---|
| AntMaze | 150,000 | 850,000 |
| ExORL   | 1,000,000 | 1,000,000 |
| Kitchen | 1,000,000 | 1,000,000 |

---

## 6. Environments and zero-shot tasks

### AntMaze (`antmaze-large-diverse-v2`)
* Max **2000** steps, start at maze center, XY discretized into a **32 x 32** grid,
  goal success when **bin distance <= 2**.
* Task families (17 tasks):
  * **goal-reaching** (5): goals `[(28,0), (0,15), (35,24), (12,24), (33,16)]`
  * **directional** (4): dot product of XY velocity with `[(-1,0), (0,1), (0,-1), (1,0)]`
  * **random-simplex** (5): seeded `opensimplex` fields (seeds 1–5)
  * **paths** (3): `path-center`, `path-loop`, `path-edges` corridors

### ExORL (RND walker / cheetah)
* Max **1000** steps. Physics features appended to the **encoder observation only**:
  * walker: `horizontal_velocity, torso_upright, torso_height`
  * cheetah: `speed`
* Per-dimension std-normalization; goal = Euclidean physics distance **< 0.1**;
  5 fixed goal states per domain.
* Velocity tasks: cheetah thresholds `{10, 1}`, walker thresholds `{0.1, 1, 4, 8}`.

### Kitchen (`kitchen-complete-v0`)
* The **7 standard D4RL subtasks** (microwave, kettle, slide, hinge, light,
  bottom burner, top burner) read directly from the sparse completion flags in
  `obs[-7:]`; no physics augmentation and no XY discretization.

---

## 7. Expected results

Normalized return `[0, 100]`; mean +/- std over **5 seeds x 20 episodes**.
Figure-5 normalization divides by the max return of any agent on a task set.

### Table 1 — aggregate per domain

| Method | antmaze-all | exorl-all | kitchen | all |
|---|---|---|---|---|
| **FRE (ours)** | **52.8 +/- 18.2** | **51.5 +/- 6.3** | **66 +/- 3** | **57 +/- 9** |
| FB | 25.8 | 43.4 | 3 | 24 |
| SF | 11.8 | 40.9 | 1 | 18 |
| GC-IQL | 40.0 | — | 59 | — |
| GC-BC | 12.0 | — | 35 | — |
| OPAL-10 | 45.6 | 28.2 | 26 | 33 |

**Sub-task breakdown (FRE):** ant-goal-reaching 48.8, ant-directional 55.2,
ant-random-simplex 21.3, ant-path-loop 67.2, ant-path-edges 60.0,
ant-path-center 64.4; exorl-walker-goals 94, exorl-cheetah-goals 58,
exorl-walker-velocity 34, exorl-cheetah-velocity 20.

> FRE uses only **32** `(state, reward)` pairs at evaluation, whereas FB/SF use
> **5120** — the paper's central efficiency claim.

### Table 4 — reward-prior ablation (AntMaze totals)

| Prior subset | Total |
|---|---|
| **FRE-all** | **47.3 +/- 7** |
| FRE-goals | 26.1 |
| FRE-lin | 31.6 |
| FRE-mlp | 25.3 |
| FRE-lin-mlp | 32.3 |
| FRE-goal-mlp | 33.8 |
| FRE-goal-lin | 46.9 |

Run an ablation with:

```bash
python -m fre.main --domain antmaze --phase all --ablation goals   # or lin, mlp, ...
```

### Figure 6 — FRE-hint

Augmenting the prior with XY-/velocity-specific functions (e.g. restricting the
linear family to position/velocity dims) improves task specificity **without any
architecture change**:

```bash
python -m fre.main --domain antmaze --phase all --hint
```

---

## 8. Validation and sanity checks

* `fre.evaluate.verify_against_table1(summary)` compares aggregate scores to the
  paper's Table 1 targets with +/- 1 std tolerance.
* Encoder/decoder sanity: the decoder MSE should converge, and predicted reward
  fields should resemble the true reward fields on held-out states
  (`EncoderTrainer.evaluate()` reports `eval_mse`, `eval_rmse`, `eval_kl`,
  `eval_z_std`). Figure 3 (qualitative visualizations) is out of scope.
* Zero-shot correctness: evaluation always encodes **exactly 32** samples and
  guarantees at least one context sample achieves the goal for goal-reaching
  tasks (`include_success_sample: true`).
* Directional / velocity tasks should beat SF; a random policy should score ~0.
* Aggregation: `mean +/- std` over 5 seeds; normalized returns in `[0, 100]`.

---

## 9. Reproduction notes / common pitfalls

* **D4RL version**: install from a commit **dated before June 2024** to avoid
  dataset/environment drift.
* **ExORL data**: use the **RND** exploratory datasets for walker/cheetah.
* **Physics augmentation**: physics features are appended to the **encoder
  observations only** — never to decoder or policy inputs.
* **Observation normalization**: per-dimension **std-normalization** for ExORL
  encoder observations.
* **AntMaze discretization**: XY is binned into a 32 x 32 grid; goal success
  uses **bin** distance `<= 2`, not continuous distance.
* **Reward discretization math**: clip to `[-1, 1]` -> rescale to `[0, 1]`
  (`(r+1)/2`) -> multiply by **32** -> `floor` -> clamp `[0, 31]`.
* **Token embedding size**: the corrected addendum is **64-d state + 64-d reward
  = 128-d**, not the erroneous `128 + 128` listed in an early appendix draft.
* **Encoder "Layers [256,256,256,256]"** denotes the transformer **MLP** dimensions
  (128 -> 256 -> 128), **not** the number of blocks; the implementation uses
  **4** blocks with 4 heads.
* **FB / SF**: do **not** reimplement — run them via `facebookresearch/controllable_agent`
  on the same RND data (SF uses ICM features).

---

## 10. References

* Original paper: *Zero-Shot Reinforcement Learning via Functional Reward Encodings*.
* Reference FRE implementation: <https://github.com/kvfrans/fre>
* D4RL: <https://github.com/Farama-Foundation/d4rl>
* ExORL: <https://github.com/denisyarats/exorl>
* FB / SF baselines: <https://github.com/facebookresearch/controllable_agent>
