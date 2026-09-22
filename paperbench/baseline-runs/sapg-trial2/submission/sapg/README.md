# SAPG: Split and Aggregate Policy Gradients

Reference implementation of **SAPG** (*Split and Aggregate Policy Gradients*), a new class of
on-policy reinforcement learning algorithms that scales to **tens of thousands of parallel
environments**.

## Core Idea

Vanilla PPO trains a single policy on one large batch of on-policy data. As the number of
parallel environments grows, the batch becomes highly redundant (the *data-duplication
problem*), and asymptotic performance saturates.

SAPG avoids this by:

1. **Splitting** the `N` parallel environments into `B` blocks. Each block `b` runs its own
   *follower* policy `pi_b` (parameters `phi_b`) and collects on-policy transitions.
2. **Aggregating** *all* transitions from *all* blocks into a single buffer. This data is
   **off-policy** with respect to any single policy.
3. Updating a **leader** policy with a PPO-style **clipped surrogate objective** using the
   importance ratio

   ```
   r_t = pi_leader(a_t | s_t) / pi_behavior(a_t | s_t),   clip(r_t, 1 - eps, 1 + eps)
   ```

   This lets the leader latch onto high-reward off-policy trajectories while retaining PPO's
   stability guarantees.

A single shared network `B_theta(obs, phi_j)` (actor) and `C_psi(obs, phi_j)` (critic) are
conditioned on per-worker embeddings `phi_j`, so all followers/leader share parameters while
remaining behaviorally distinct.

## Repository Layout

```
sapg/
├── main.py                      # Entry point: train / eval CLI
├── sapg/
│   ├── networks.py              # Shared conditioned B_theta / C_psi (MLP + LSTM)
│   ├── policy.py                # Gaussian policy, learnable sigma, per-worker sigma option
│   ├── sapg_algorithm.py        # Follower/leader updates, clipped surrogate, aggregation
│   ├── rollout_buffer.py        # Block-wise storage, off-policy aggregation, GAE
│   ├── ppo_baseline.py          # Vanilla PPO baseline
│   ├── curriculum.py            # Success-tolerance curriculum for hard tasks
│   ├── eval.py                  # Success / episode-reward evaluation
│   ├── utils.py                 # Logging, seeding, checkpointing, config I/O
│   ├── envs/
│   │   ├── allegro_kuka.py      # Regrasping / Throw / Reorientation (23 DoF)
│   │   ├── shadow_hand.py       # 24-DoF in-hand reorientation
│   │   ├── allegro_hand.py      # 16-DoF in-hand reorientation
│   │   └── reward.py            # r_reach, r_lift, r_target, r_success, r_orientation
│   └── configs/
│       ├── allegro_kuka.yaml    # Table 2 hyperparameters
│       ├── shadow_hand.yaml     # Table 3 hyperparameters
│       └── allegro_hand.yaml    # Table 4 hyperparameters
└── experiments/
    ├── run_batchsize_sweep.py   # Figure 2: performance vs. batch size
    ├── run_ablations.py         # Figure 6: aggregation / architecture ablations
    └── run_reconstruction.py    # Figure 8: L2 reconstruction vs. network size
```

## Installation

```bash
pip install -r requirements.txt
```

Required: `torch`, `numpy`, `PyYAML`, `tensorboard`.
Optional: `wandb` (logging), `isaacgym` (GPU-parallel simulation), `mujoco`/`gym` (CPU fallback).

> **Simulation backends.** The environments are *simulator-agnostic*. If IsaacGym is available
> it is used for massively parallel GPU simulation; otherwise a lightweight, fully vectorized
> analytic fallback simulator (pure PyTorch) is used so the full pipeline runs on CPU or a
> single GPU. Scaling behavior at tens of thousands of envs requires IsaacGym.

## Quick Start

### Train SAPG on AllegroKuka (Regrasping)

```bash
python main.py --algo sapg --env allegro_kuka --task regrasping \
    --num_envs 8192 --num_blocks 8 --aggregation leader \
    --total_steps 100000000 --log_dir runs/allegro_kuka_regrasping
```

### Train vanilla PPO (baseline)

```bash
python main.py --algo ppo --env allegro_kuka --task regrasping \
    --num_envs 8192 --total_steps 100000000 --log_dir runs/ppo_allegro_kuka
```

### Load a YAML config (CLI flags take precedence)

```bash
python main.py --config sapg/configs/shadow_hand.yaml --algo sapg
```

### Evaluate a checkpoint

```bash
python main.py --eval --algo sapg --env shadow_hand \
    --checkpoint runs/shadow_hand/checkpoint.pt --eval_episodes 32
```

## Hyperparameters (Tables 2–4)

| Setting                  | AllegroKuka (Table 2) | ShadowHand (Table 3) | AllegroHand (Table 4) |
|--------------------------|-----------------------|----------------------|-----------------------|
| Mean net                 | LSTM (1×768)          | MLP 512-512-256-128  | MLP 512-256-128       |
| Pre-MLP (LSTM input)     | 768-512-256 (ELU)     | —                    | —                     |
| Activation               | ELU                   | ELU                  | ELU                   |
| Horizon                  | 16                    | 8                    | 8                     |
| Mini-epochs              | 2                     | 5                    | 5                     |
| Mini-batch size          | `num_envs × 4`        | `num_envs × 4`       | `num_envs × 4`        |
| `gamma` / `tau`          | 0.99 / 0.95           | 0.99 / 0.95          | 0.99 / 0.95           |
| `clip_eps`               | 0.2                   | 0.2                  | 0.2                   |
| `value_loss_coef`        | 1.0                   | 1.0                  | 1.0                   |
| `bounds_loss_coef`       | 0.001                 | 0.001                | 0.001                 |
| `entropy_coef`           | 0.0                   | 0.0                  | 0.0                   |
| `lr`                     | 3e-4                  | 3e-4                 | 3e-4                  |
| `kl_threshold`           | 0.016                 | 0.016                | 0.016                 |
| Success tolerance `delta`| 7.5 cm → 1 cm         | 10 cm                | 10 cm                 |

**KL-based LR adaptation:** if the approximate KL exceeds `1.5 × kl_threshold`, halve the LR;
if it falls below `kl_threshold / 1.5`, double the LR.

**Loss:** `clipped_surrogate − value_loss_coef × value_loss + bounds_loss_coef × bounds_loss`
(entropy coefficient is 0 in the default configs).

## Environments

### AllegroKuka (hard tasks)

Allegro 16-DoF hand + Kuka 7-DoF arm (23 joints). Observation:

```
o_t = [q, q_dot, x_t, v_t, omega_t, g_t, z_t]
```

with `q, q_dot ∈ R^23`, object pose `x_t ∈ R^7`, object linear/angular velocity
`v_t, omega_t ∈ R^3`, task goal `g_t`, and auxiliary features `z_t`.

Tasks:
- **Regrasping** — `g_t ∈ R^3`; requires holding the object at the goal for `K = 30` steps.
- **Throw** — bucket target placed out of reach.
- **Reorientation** — `g_t ∈ R^7` (target pose).

Reward: `w1·r_reach + w2·r_lift + w3·r_target + w4·r_success`.
Success: `||g_t − (x_t)_{0:3}|| ≤ delta`.

**Curriculum:** whenever the average number of successes per episode exceeds 3, the tolerance
`delta` is reduced by 10% (7.5 cm → 1 cm).

### ShadowHand / AllegroHand (easy tasks)

24-DoF and 16-DoF in-hand cube reorientation. Goal is a target quaternion `g_t ∈ R^4`.
Reward = dense orientation error + sparse success bonus. The reported metric is **net episode
reward** (compared against PQL, Li et al. 2023).

## Reproducing the Paper's Figures

### Figure 2 — Performance vs. batch size

```bash
python experiments/run_batchsize_sweep.py \
    --env allegro_kuka --task regrasping \
    --batch_sizes 256 512 1024 2048 4096 8192 16384 \
    --algos ppo sapg --seeds 0 1 2 \
    --output results/batchsize_sweep.json
```

Expected: PPO's asymptotic performance saturates as the batch grows, while SAPG continues to
improve (SAPG curve exceeds the PPO plateau).

### Hard tasks (AllegroKuka)

```bash
for task in regrasping throw reorientation; do
  python main.py --algo sapg --env allegro_kuka --task $task \
      --num_envs 8192 --num_blocks 8 --log_dir runs/sapg_$task
  python main.py --algo ppo  --env allegro_kuka --task $task \
      --num_envs 8192 --log_dir runs/ppo_$task
done
```

Metric: successes per episode. SAPG should reach significantly higher asymptotic success than
vanilla PPO (which fails to obtain positive success on some tasks).

### Easy tasks (ShadowHand / AllegroHand)

```bash
python main.py --algo sapg --env shadow_hand  --num_envs 8192 --num_blocks 8
python main.py --algo sapg --env allegro_hand --num_envs 8192 --num_blocks 8
```

Metric: net episode reward.

### Figure 6 — Ablations

```bash
python experiments/run_ablations.py \
    --env allegro_kuka --task regrasping \
    --modes leader symmetric no_shared_net per_worker_sigma \
    --seeds 0 1 2 --output results/ablations.json
```

- `leader` — default SAPG (one designated leader updated on the union of all blocks' data).
- `symmetric` — no designated leader; each worker is updated with all *other* workers'
  off-policy data.
- `no_shared_net` — separate networks per worker (no parameter sharing).
- `per_worker_sigma` — entropy-exploration variant where each block owns its own learnable
  sigma vector.

Expected: leader-based aggregation outperforms symmetric aggregation.

### Figure 8 — Representation / reconstruction study

```bash
python experiments/run_reconstruction.py \
    --sizes 16 32 64 128 256 512 1024 \
    --num_workers 8 --steps 400000 \
    --output results/reconstruction.json
```

Two-layer ReLU networks of equal size (conditioned vs. unconditioned), trained with the Adam
optimizer (PyTorch defaults) on 400k state-transitions using an L2 reconstruction loss.
Reproduces reconstruction error vs. network size.

## Sanity Checks

The implementation includes the following verifiable behaviors:

- **Importance ratios** — the leader update computes `pi_leader / pi_behavior` against the
  stored behavior log-probs and clips to `[1 − eps, 1 + eps]`.
- **KL-based LR adaptation** — LR halves/doubles at the `1.5×` thresholds.
- **GAE correctness** — advantages use `gamma = 0.99`, `tau = 0.95` with per-block bootstrapping.
- **Full aggregation** — every transition from every block contributes to each leader update
  (`RolloutBuffer.aggregate()`).

## Logging & Checkpoints

- Metrics are written to stdout, a JSONL file, and TensorBoard (`--use_tensorboard`).
- Optional Weights & Biases logging via `--use_wandb --wandb_project sapg`.
- Checkpoints are saved every `--save_interval` updates and can be reloaded with
  `--checkpoint <path>`.

## Notes on Ambiguities

- Exact reward weights (`w1..w4`) and the embedding size `phi_dim` are not fully specified in
  the paper; reasonable defaults (IsaacGymEnvs conventions, `phi_dim = 8`) are used and exposed
  via the YAML configs.
- If IsaacGym is unavailable, the analytic fallback simulator is used with reduced parallelism;
  absolute wall-clock scaling differs, but algorithmic behavior is preserved.
