# SAPG: Split and Aggregate Policy Gradients

Reference implementation of **SAPG** (Split and Aggregate Policy Gradients), a new class of
on-policy reinforcement learning algorithms that scale to tens of thousands of parallel
environments.

## Overview

SAPG splits a large batch of parallel environments into **blocks**. Each block is optimized by
its own **follower** policy (standard on-policy PPO on its own rollout data). All transitions
produced by all followers are then **aggregated** and used to train a single **leader** policy
with PPO's clipped surrogate objective (with off-policy importance weighting). The leader is the
deployed policy; the followers exist only to generate diverse data. Because every transition is
reused by the leader, no data is wasted, and the effective batch size seen by the leader grows
with the number of blocks.

```
                 ┌────────────┐   ┌────────────┐        ┌────────────┐
   env block 0 → │ follower 0 │   │ follower 1 │  ...   │ follower K │  ← per-block PPO
                 └─────┬──────┘   └─────┬──────┘        └─────┬──────┘
                       │                │                     │
                       └────────────────┴──────────┬──────────┘
                                                   ▼
                                          ┌─────────────────┐
                                          │  aggregate all  │  ← off-policy data
                                          │  transitions    │
                                          └────────┬────────┘
                                                   ▼
                                          ┌─────────────────┐
                                          │     leader      │  ← deployed policy
                                          │  (PPO clipped)  │
                                          └─────────────────┘
```

## Repository layout

```
sapg/
├── main.py                      # Entry point: train/eval orchestration
├── configs/
│   ├── default.yaml             # Base hyperparameters
│   ├── allegrokuka.yaml         # AllegroKuka (regrasping / throw / reorientation)
│   ├── shadow_hand.yaml         # Shadow Hand (24-DoF)
│   └── allegro_hand.yaml        # Allegro Hand (16-DoF)
├── sapg/
│   ├── networks.py              # Shared B_theta / C_psi conditioned on phi_j
│   ├── follower.py              # Per-block follower PPO update
│   ├── leader.py                # Leader aggregation + off-policy PPO update
│   ├── buffer.py                # Rollout buffer + aggregated (off-policy) buffer
│   ├── ppo.py                   # Clipped surrogate, GAE, KL-LR, loss assembly
│   └── aggregation.py           # Leader / symmetric aggregation variants
├── envs/
│   ├── allegrokuka.py           # Regrasping / Throw / Reorientation + curriculum
│   ├── shadow_hand.py           # In-hand reorientation (24-DoF)
│   ├── allegro_hand.py          # In-hand reorientation (16-DoF)
│   └── wrappers.py              # Obs normalization, block assignment, curriculum
├── utils/
│   ├── curriculum.py            # Success-tolerance curriculum (7.5cm → 1cm, −10%)
│   ├── logger.py                # Metrics: successes, episode reward, PPO diagnostics
│   └── checkpoint.py            # Save/load training state
├── experiments/
│   ├── run_batchsize_sweep.py   # Figure 2: PPO vs SAPG vs batch size
│   ├── run_ablation.py          # Figure 6: symmetric aggregation, entropy variants
│   └── run_reconstruction.py    # Figure 8: 2-layer MLP L2 reconstruction
├── tests/
│   ├── test_ppo.py
│   ├── test_aggregation.py
│   └── test_curriculum.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

The base install is CPU-runnable: the environments ship with a framework-agnostic NumPy
surrogate simulator that preserves the observation layout, action space, reward structure, and
curriculum behaviour of the original IsaacGym / MuJoCo tasks. To use the real physics backends,
uncomment `isaacgym` (AllegroKuka) and/or `mujoco` + `gymnasium` (Shadow Hand / Allegro Hand) in
`requirements.txt`.

## Quick start

Train SAPG on AllegroKuka regrasping:

```bash
python main.py --config configs/allegrokuka.yaml --task regrasping
```

Train on Shadow Hand:

```bash
python main.py --config configs/shadow_hand.yaml
```

Evaluate a checkpoint:

```bash
python main.py --config configs/allegrokuka.yaml --task throw --eval --checkpoint runs/allegrokuka/ckpt.pt
```

Common CLI overrides (see `python main.py --help`):

| Flag | Meaning |
| --- | --- |
| `--config` | Path to a YAML config (defaults to `configs/default.yaml`) |
| `--task` | Task name for AllegroKuka (`regrasping` / `throw` / `reorientation`) |
| `--num-envs` | Number of parallel environments (batch size) |
| `--num-blocks` | Number of follower blocks |
| `--iterations` | Number of training iterations |
| `--seed` | Random seed |
| `--log-dir` | Directory for TensorBoard / JSONL logs |
| `--eval` | Run evaluation only |
| `--checkpoint` | Checkpoint path to load/save |

## Algorithm

### Networks (`sapg/networks.py`)

A single shared actor `B_theta` (Gaussian mean) and critic `C_psi` are conditioned on a
per-follower / per-leader parameter vector `phi_j` (concatenated to the observation). This lets
all followers and the leader share weights while remaining distinguishable.

* **AllegroKuka**: MLP trunk `[768, 512, 256]` with ELU, followed by a 1-layer LSTM with 768
  hidden units. Gaussian sigma is a fixed learnable vector (input-independent).
* **Shadow Hand**: MLP `[512, 512, 256, 128]`, ELU.
* **Allegro Hand**: MLP `[512, 256, 128]`, ELU.
* **Entropy-exploration ablation**: each block gets its own learnable sigma vector
  (`network.per_block_sigma: true`).

### Follower update (`sapg/follower.py`)

Each block runs standard on-policy PPO on its own rollout:

1. Collect a horizon of transitions.
2. Compute GAE advantages (`gamma=0.99`, `tau=0.95`).
3. Update the shared networks with the clipped surrogate objective, value loss
   (`critic_coeff=4.0`), and bounds loss (`bounds_loss_coeff=1e-4`).

### Leader aggregation (`sapg/leader.py`, `sapg/aggregation.py`)

All transitions from all followers are concatenated (`AggregatedBuffer`) and used to update the
leader. Off-policy correction uses importance weights
`w = exp(log pi_leader(a|s) - log pi_follower(a|s))`, clamped at `max_importance_weight`
(default 10.0), and the PPO ratio is anchored on the behaviour (follower) log-probs.

The **symmetric aggregation** ablation (`sapg.mode: symmetric`) removes the designated leader:
every worker is updated on the union of all other workers' data.

### PPO core (`sapg/ppo.py`)

* Clipped surrogate: `L = min(r·A, clip(r, 1−eps, 1+eps)·A)`
* GAE-lambda advantage estimation
* Adaptive learning rate via KL threshold (`kl_threshold=0.016`)
* Gradient norm clipping (`grad_norm_clip=1.0`)
* Entropy coefficient `0.0` by default
* Mini-batch size = `num_envs * 4`

### Curriculum (`utils/curriculum.py`)

Success-tolerance curriculum for the hard AllegroKuka tasks: the tolerance `delta` starts at
**7.5 cm** and is multiplied by **0.9** (down to a **1 cm** floor) whenever the average number of
successes per episode exceeds **3**. After each success the target/object is resampled at a random
location.

## Hyperparameters

| | AllegroKuka | Shadow Hand | Allegro Hand |
| --- | --- | --- | --- |
| `lr` | 1e-4 | 5e-4 | 5e-4 |
| `clip_eps` | 0.1 | 0.1 | 0.2 |
| `horizon` | 16 | 8 | 8 |
| `mini_epochs` | 2 | 5 | 5 |
| `gamma` | 0.99 | 0.99 | 0.99 |
| `tau` | 0.95 | 0.95 | 0.95 |
| `kl_threshold` | 0.016 | 0.016 | 0.016 |
| `grad_norm_clip` | 1.0 | 1.0 | 1.0 |
| `entropy_coeff` | 0.0 | 0.0 | 0.0 |
| `critic_coeff` | 4.0 | 4.0 | 4.0 |
| `bounds_loss_coeff` | 1e-4 | 1e-4 | 1e-4 |
| `mini_batch_size` | `num_envs*4` | `num_envs*4` | `num_envs*4` |
| `lstm_seq_len` | 16 | — | — |

## Experiments

### Figure 2 — Batch-size scaling

```bash
python -m experiments.run_batchsize_sweep \
    --config configs/allegrokuka.yaml --task regrasping \
    --batch-sizes 1024 4096 16384 65536 \
    --algorithms ppo sapg --num-blocks 8 --iterations 2000 \
    --output results/batchsize.json
```

Expected: PPO performance saturates beyond a certain batch size, while SAPG continues to improve
and reaches a higher asymptotic performance.

### Main results on hard tasks (AllegroKuka)

```bash
for task in regrasping throw reorientation; do
  python main.py --config configs/allegrokuka.yaml --task $task --num-envs 4096 --num-blocks 8
done
```

Metric = successes per episode. Expected: SAPG achieves significantly higher asymptotic success
than PPO; PPO may fail to obtain positive success on some tasks. The curriculum `delta` should
decrease as average successes exceed 3.

### Easy tasks (Shadow Hand / Allegro Hand)

```bash
python main.py --config configs/shadow_hand.yaml
python main.py --config configs/allegro_hand.yaml
```

Metric = net episode reward. Expected: SAPG matches or exceeds PPO/PQL baselines.

### Figure 6 — Ablation

```bash
python -m experiments.run_ablation \
    --config configs/allegrokuka.yaml --task regrasping \
    --variants leader symmetric entropy \
    --output results/ablation.json
```

Variants:

* `leader` — default SAPG (leader-based aggregation).
* `symmetric` — no designated leader; each worker updated on the union of all data.
* `entropy` — per-block learnable sigma vectors for exploration.

Expected: leader-based SAPG outperforms symmetric aggregation.

### Figure 8 — Reconstruction

```bash
python -m experiments.run_reconstruction \
    --num-transitions 400000 --sizes 16 32 64 128 256 512 \
    --output results/reconstruction.json
```

Trains 2-layer ReLU MLPs of increasing hidden size with Adam (PyTorch defaults) and L2
reconstruction loss on 400k state-transitions, demonstrating that modest networks can reconstruct
transition data.

## Tests

```bash
pytest tests/ -v
# or run each suite directly:
python tests/test_ppo.py
python tests/test_aggregation.py
python tests/test_curriculum.py
```

The tests validate:

* PPO clipped surrogate objective and GAE against hand-computed reference values.
* KL-based adaptive learning-rate scheduling and clamping.
* Aggregation uses **all** transitions from **all** followers (no data waste).
* Importance weights are finite, non-negative, and clamped.
* Symmetric aggregation updates every worker.
* Follower diversity: distinct `phi_j` produce distinct action distributions.
* Curriculum triggers correctly (delta decrements at the threshold, floor clamping, EMA, warmup).

## Reproducing the paper

1. Install dependencies: `pip install -r requirements.txt`.
2. Run the batch-size sweep (Figure 2) and confirm SAPG exceeds the PPO plateau.
3. Run the hard-task experiments (AllegroKuka) and confirm SAPG obtains positive success where
   PPO fails.
4. Run the easy-task experiments (Shadow Hand, Allegro Hand) and confirm SAPG matches/exceeds
   baselines.
5. Run the ablation (Figure 6) and confirm leader-based SAPG beats symmetric aggregation.
6. Run the reconstruction experiment (Figure 8).

All runs write TensorBoard event files and a `metrics.jsonl` to the configured `--log-dir`.

## Notes on fidelity

The paper does not fully specify every reward weight or the exact number of blocks. Where values
are unspecified we use reasonable defaults and expose them as config keys:

* Reward weights: `env.w_reach`, `env.w_lift`, `env.w_target`, `env.w_success`, `env.w_orientation`.
* Number of blocks / followers: `env.num_blocks` (default 8).
* Off-policy importance weighting: PPO ratio anchored on follower log-probs, clamped at
  `sapg.max_importance_weight` (default 10.0).
* Entropy variant: per-block learnable sigma vectors (`network.per_block_sigma`).

The bundled environments use an analytic surrogate simulator so the full pipeline runs on CPU
without IsaacGym/MuJoCo; the observation layout, action space, reward structure, and curriculum
semantics follow the paper's specification.
