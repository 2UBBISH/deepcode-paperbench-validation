# SAPG: Split and Aggregate Policy Gradients

Reference implementation of **SAPG** (*Split and Aggregate Policy Gradients*), a new class
of on-policy reinforcement learning algorithms that scale to tens of thousands of parallel
environments.

Instead of running a single PPO policy across all environments, SAPG:

1. **SPLIT** — divides the `num_envs` parallel environments into `B` blocks. Each block `b`
   runs its own *follower* policy `pi_{phi_b}` (a shared network `B_theta` conditioned on a
   per-worker embedding `phi_b`).
2. **AGGREGATE** — designates one worker as the *leader*. The leader is updated using **all**
   off-policy data collected by every follower, via an importance-weighted / clipped PPO
   surrogate. Followers are updated on their own on-policy data.

This increases data diversity and asymptotic performance over vanilla PPO at large batch
sizes, where PPO is known to saturate.

---

## Repository layout

```
sapg/
├── main.py                      # Entry point: train SAPG/PPO on a task
├── configs/
│   ├── default.yaml             # Global config (num_envs, blocks, PPO/SAPG hyperparams)
│   ├── allegro_kuka.yaml        # Hyperparams Table 2 (LSTM actor)
│   ├── shadow_hand.yaml         # Hyperparams Table 3 (MLP actor)
│   └── allegro_hand.yaml        # Hyperparams Table 4 (MLP actor)
├── sapg/
│   ├── __init__.py              # Public API re-exports
│   ├── networks.py              # Actor B_theta(phi_j), Critic C_psi(phi_j)
│   ├── policy.py                # Gaussian policy, per-block sigma, bounds loss
│   ├── sapg_algorithm.py        # Split + Aggregate update (core contribution)
│   ├── ppo.py                   # Baseline PPO (clipped surrogate)
│   ├── rollout_buffer.py        # On/off-policy storage, LSTM sequences
│   ├── aggregation.py           # Leader/follower aggregation logic
│   └── utils.py                 # GAE, KL-LR, grad clip, ELU helpers
├── envs/
│   ├── __init__.py              # Env registry + make_env factory
│   ├── allegro_kuka.py          # Regrasping / Throw / Reorientation
│   ├── shadow_hand.py           # In-hand reorientation (24-DoF)
│   ├── allegro_hand.py          # In-hand reorientation (16-DoF)
│   └── curriculum.py            # Success-tolerance curriculum
├── experiments/
│   ├── run_sapg.py              # SAPG training runs
│   ├── run_ppo_baseline.py      # PPO batch-size sweep (Fig 2)
│   ├── run_ablations.py         # Symmetric aggregation etc. (Fig 6)
│   └── run_reconstruction.py    # L2 reconstruction experiment (Fig 8)
├── eval/
│   ├── evaluate.py              # Successes metric / episode reward
│   └── plot.py                  # Reproduce figures
├── README.md
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `torch>=1.13`, `numpy`, `scipy`, `PyYAML`, `matplotlib`.

**Simulation backends.** The paper uses [IsaacGym](https://developer.nvidia.com/isaac-gym)
(Makoviychuk et al., 2021) for GPU-parallel AllegroKuka / ShadowHand / AllegroHand
environments, which requires an NVIDIA GPU. IsaacGym is not distributed on PyPI and must be
installed manually following NVIDIA's instructions.

For CPU-only reproduction and debugging, this repository ships **NumPy-based kinematic proxy
environments** (`envs/allegro_kuka.py`, `envs/shadow_hand.py`, `envs/allegro_hand.py`) that
expose the same `reset` / `step` / `get_obs_dim` / `get_action_dim` interface. They preserve
the observation layout, reward structure, success metric, and curriculum of the paper so the
full SAPG/PPO stack can be exercised end-to-end without a GPU. MuJoCo 3.0 is an optional
alternative backend.

---

## Quick start

Train SAPG on AllegroKuka Regrasping:

```bash
python main.py --config configs/allegro_kuka.yaml --algo sapg --task regrasping
```

Train vanilla PPO (single block) on the same task:

```bash
python main.py --config configs/allegro_kuka.yaml --algo ppo --task regrasping
```

Useful CLI flags (see `python main.py --help`):

| Flag | Meaning |
| --- | --- |
| `--config` | Path to a YAML config (task configs inherit `default.yaml`) |
| `--algo` | `sapg` or `ppo` |
| `--env_name` | `allegro_kuka`, `shadow_hand`, `allegro_hand` |
| `--task` | `regrasping`, `throw`, `reorientation` |
| `--num_envs` | Number of parallel environments (batch size) |
| `--num_workers` | Number of blocks `B` (followers + leader) |
| `--horizon` | Rollout horizon per update |
| `--max_iterations` | Total training iterations |
| `--aggregation_mode` | `leader`, `symmetric`, or `none` |
| `--device` | `cuda` or `cpu` |
| `--output_dir` | Where checkpoints / history are written |

> **Sanity check:** setting `--num_workers 1` (or `aggregation_mode: none`) reduces SAPG to
> standard PPO.

---

## Algorithm summary

### Networks (`sapg/networks.py`)

A single shared actor `B_theta` and critic `C_psi` are conditioned on a per-worker embedding
`phi_j` (a learnable parameter vector indexed by worker id). Only `phi_j` differs between
followers/leader, so the parameter count is constant in the number of workers.

* **AllegroKuka** — `MLP(768x512x256, ELU) -> LSTM(1 layer, 768 hidden) -> mean head`,
  with an input-independent learnable `sigma` vector.
* **ShadowHand** — `MLP(512x512x256x128, ELU) -> mean head`.
* **AllegroHand** — `MLP(512x256x128, ELU) -> mean head`.

### Policy (`sapg/policy.py`)

Diagonal Gaussian policy `a ~ N(mean, sigma)`. `sigma` is a fixed learnable vector
(input-independent). When `use_entropy_exploration: true`, **each block** gets its own
learnable `sigma` vector. Optional tanh squashing with a bounds penalty
(`bounds_loss_coef = 1e-4`).

### Split and Aggregate (`sapg/sapg_algorithm.py`, `sapg/aggregation.py`)

Per update:

1. **SPLIT** — partition `num_envs` into `B` blocks; each block `b` collects a
   horizon-length rollout with follower `pi_{phi_b}` (on-policy for that follower).
2. **AGGREGATE** — the leader is updated using **all** off-policy data from all followers
   with the importance ratio

   ```
   r = pi_leader(a | s) / pi_follower(a | s)
   ```

   and the PPO clipped surrogate

   ```
   L = E[ min( r * A, clip(r, 1 - eps, 1 + eps) * A ) ]
   ```

   Followers are updated on their own on-policy data.

Key hyperparameters (paper defaults):

| Parameter | Value |
| --- | --- |
| `gamma` | 0.99 |
| `tau` (GAE) | 0.95 |
| `critic_coef` (lambda') | 4.0 |
| `entropy_coef` | 0.0 |
| `max_grad_norm` | 1.0 |
| `kl_target` (adaptive LR) | 0.016 |
| `mini_batch_size` | `num_envs * 4` |
| `clip_epsilon` | 0.1 (AllegroKuka / ShadowHand), 0.2 (AllegroHand) |

### Aggregation modes (`sapg/aggregation.py`)

* `leader` — one designated leader updated on all followers' off-policy data (SAPG).
* `symmetric` — no leader; every worker is updated with off-policy data from all other
  workers symmetrically (ablation, Fig 6).
* `none` — independent PPO per block (reduces to PPO when `B = 1`).

---

## Environments

Observation layout (all tasks):

```
o_t = [ q, q_dot, x_t, v_t, omega_t, g_t, z_t ]
```

* **AllegroKuka** — Allegro 16-DoF hand + Kuka 7-DoF arm = 23 joints.
  * *Regrasping* — goal `g_t` in R^3; success requires the object within tolerance for
    `K = 30` consecutive steps.
  * *Throw* — bucket goal.
  * *Reorientation* — goal pose `g_t` in R^7.
  * Reward: `w1 * r_reach + r_lift + r_target + r_success`.
  * Metric: **successes per episode**.
* **ShadowHand** — 24-DoF in-hand cube reorientation, goal quaternion `g_t` in R^4.
  Reward = orientation error + success bonus. Metric: **net episode reward**.
* **AllegroHand** — 16-DoF in-hand reorientation, same structure as ShadowHand.

### Curriculum (`envs/curriculum.py`)

Success tolerance `delta` starts at **7.5 cm** and decays by **10%** (multiplicatively)
whenever the mean successes per episode exceeds **3**, down to a minimum of **1 cm**.

---

## Reproducing the paper's experiments

### Experiment 1 — PPO batch-size saturation (Figure 2)

Sweep PPO across increasing batch sizes and overlay the SAPG asymptote:

```bash
python -m experiments.run_ppo_baseline \
    --env_name allegro_kuka --task regrasping \
    --batch_sizes 512,1024,2048,4096,8192 \
    --output_dir runs/ppo_baseline

python -m experiments.run_sapg \
    --env_name allegro_kuka --task regrasping \
    --output_dir runs/sapg

python -m eval.plot --figure fig2 \
    --ppo runs/ppo_baseline/ppo_baseline_summary.json \
    --sapg runs/sapg/sapg_summary.json \
    --task regrasping
```

**Expected:** PPO performance saturates beyond a certain batch size; SAPG exceeds the PPO
asymptote.

### Experiment 2 — Main SAPG results

```bash
python -m experiments.run_sapg --env_name allegro_kuka --task regrasping
python -m experiments.run_sapg --env_name allegro_kuka --task throw
python -m experiments.run_sapg --env_name allegro_kuka --task reorientation
python -m experiments.run_sapg --env_name shadow_hand  --task reorientation
python -m experiments.run_sapg --env_name allegro_hand --task reorientation
```

**Expected:** significantly higher asymptotic performance than PPO; positive success where
vanilla PPO fails to achieve any.

### Experiment 3 — Ablations (Figure 6)

```bash
python -m experiments.run_ablations \
    --env_name allegro_kuka --task regrasping \
    --ablation mode \
    --modes leader,symmetric,none \
    --output_dir runs/ablations

python -m eval.plot --figure fig6 \
    --ablation runs/ablations/ablation_summary.json \
    --task regrasping
```

**Expected:** leader-based SAPG outperforms symmetric aggregation.

### Experiment 4 — L2 reconstruction (Figure 8)

Two-layer networks with equal hidden size per layer, ReLU activation, Adam (PyTorch
defaults), trained on 400k state-transitions with L2 reconstruction loss:

```bash
python -m experiments.run_reconstruction \
    --num_transitions 400000 \
    --sizes 16,32,64,128,256 \
    --methods shared,per_worker \
    --plot
```

**Expected:** reconstruction error vs. network size curves for shared vs. per-worker
networks.

---

## Evaluation

Evaluate a trained checkpoint:

```bash
python -m eval.evaluate \
    --checkpoint runs/sapg/checkpoint_final.pt \
    --config configs/allegro_kuka.yaml \
    --algo sapg \
    --num_episodes 100
```

Reported metrics:

* `successes_per_episode` — primary metric for hard tasks (regrasping / throw / reorientation).
* `mean_episode_reward` — primary metric for easier tasks.
* `success_rate`, `mean_episode_length`, `tolerance`.

---

## Verification checkpoints

The implementation includes the following correctness checks:

* **Importance-sampling ratio & clipping** — `Aggregator.clipped_surrogate` computes
  `r = exp(new_log_probs - old_log_probs)` and clips to `[1 - eps, 1 + eps]`.
* **GAE** — `utils.compute_gae` matches hand-computed values with `gamma = 0.99`,
  `tau = 0.95`.
* **Curriculum** — tolerance decrements by `decay` only when mean successes exceed the
  threshold (default 3), clamped at the minimum.
* **Sanity** — a single-block SAPG (`num_workers = 1`) reduces to standard PPO.

---

## Configuration

Configs are plain YAML. Task configs inherit from `configs/default.yaml` via a
`defaults: [default]` directive, so only task-specific overrides need to be declared.
CLI flags override config values.

Key config groups:

* **Environment / parallelism** — `num_envs`, `num_workers`, `horizon`, `max_iterations`.
* **Optimization** — `gamma`, `tau`, `clip_epsilon`, `critic_coef`, `entropy_coef`,
  `max_grad_norm`, `learning_rate`, `num_learning_epochs`, `mini_batch_size`, `kl_target`.
* **Network** — `hidden_dims`, `critic_hidden_dims`, `embed_dim`, `activation`,
  `use_lstm`, `lstm_hidden`, `lstm_layers`, `seq_len`, `condition_mode`.
* **Aggregation** — `aggregation_mode`, `leader_id`, `rotate_leader`,
  `use_entropy_exploration`.
* **Curriculum** — `use_curriculum`, `curriculum_initial_tolerance`,
  `curriculum_min_tolerance`, `curriculum_decay`, `curriculum_success_threshold`.
* **Logging** — `log_interval`, `save_interval`, `eval_interval`, `output_dir`, `wandb`.

---

## Notes on defaults

A few details are not fully specified in the paper; this implementation exposes them as
config options with reasonable defaults:

* **Reward weights** (`w1`, `r_lift_w`, `r_target_w`, `r_success_w`) — default to
  `1.0, 1.0, 1.0, 10.0`.
* **Leader selection** — fixed designated leader (`leader_id: 0`) by default; rotation is
  opt-in via `rotate_leader: true`.
* **Importance sampling** — PPO-style clipped ratio with `clip_epsilon` from config
  (0.1 for AllegroKuka / ShadowHand, 0.2 for AllegroHand).
* **Entropy exploration** — per-block learnable `sigma` when
  `use_entropy_exploration: true`.

---

## Citation

If you use this code, please cite the SAPG paper:

```bibtex
@inproceedings{sapg,
  title     = {SAPG: Split and Aggregate Policy Gradients},
  author    = {Anonymous},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024}
}
```
