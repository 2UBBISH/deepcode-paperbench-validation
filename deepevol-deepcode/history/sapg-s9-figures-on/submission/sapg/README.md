# SAPG: Split and Aggregate Policy Gradients

Reference implementation of **SAPG** (Split and Aggregate Policy Gradients), an
on-policy reinforcement learning algorithm that scales to tens of thousands of
parallel environments by:

1. **Splitting** `N` parallel environments into `M` blocks of `N/M` environments.
2. Training `M` diverse policies — **1 leader** + `M-1` **followers** — that share a
   backbone network (`B_theta` actor, `C_psi` critic) conditioned on per-policy
   latent parameters `phi_j`.
3. **Aggregating** follower data into the leader via importance-sampled off-policy
   PPO updates with a `mu` correction term.

This repository reproduces the paper's Table 1, Figures 5–8, and the ablation
study in Figure 6.

---

## Repository layout

```
sapg/
├── main.py                      # CLI entry point (train / baseline / ablate / diversity / plot / eval)
├── sapg/
│   ├── algorithm.py             # SAPG core: Algorithm 1 loop, leader/follower updates
│   ├── losses.py                # L_on (Eq.2), L_off (Eq.3), combined (Eq.4), critic (Eq.5-9)
│   ├── models.py                # Shared actor B_theta + critic C_psi, phi_j conditioning
│   ├── rollout.py               # Block partitioning, per-policy buffers, off-policy subsampling
│   ├── ppo.py                   # PPO baseline (single policy)
│   ├── gae.py                   # Advantage estimation (GAE tau=0.95), n-step returns (n=3)
│   ├── entropy.py               # Follower entropy regularization sigma*(i-1)*H(pi)
│   └── utils.py                 # KL-adaptive LR, grad clipping, logging, seeding
├── envs/
│   ├── isaacgym_wrapper.py      # IsaacGym env wrapper, obs construction o_t
│   ├── allegrokuka.py           # Regrasping / Throw / Reorientation tasks
│   ├── shadow_hand.py           # 24-DoF in-hand reorientation
│   └── allegro_hand.py          # 16-DoF in-hand reorientation
├── configs/
│   ├── allegrokuka.yaml         # Table 2 hyperparameters
│   ├── shadowhand.yaml          # Table 3 hyperparameters
│   └── allegrohand.yaml         # Table 4 hyperparameters
├── experiments/
│   ├── train_sapg.py            # Full SAPG training
│   ├── train_baselines.py       # PPO / PBT / PQL baselines
│   ├── ablations.py             # symmetric, w/o off-policy, high off-policy ratio, entropy coefs
│   ├── diversity_metrics.py     # PCA (Fig.7) + MLP reconstruction (Fig.8)
│   └── plot_results.py          # Reproduce Fig.5, Fig.6, Table 1
├── README.md
└── requirements.txt
```

---

## Setup

### 1. Python environment

Python 3.8–3.10 is required.

```bash
cd sapg
pip install -r requirements.txt
```

### 2. IsaacGym (optional but recommended)

IsaacGym is **not pip-installable**. Download *IsaacGym Preview 4* from the NVIDIA
developer portal, then:

```bash
cd isaacgym/python
pip install -e .
```

Requirements: NVIDIA GPU + CUDA 11.x.

### 3. CPU-only / mock mode

If IsaacGym is unavailable, every script supports a pure-PyTorch analytic
dynamics fallback via `--force-mock`. This is intended for pipeline debugging and
CI, **not** for reproducing paper numbers.

```bash
python main.py train --task allegrohand --num-envs 1024 --num-policies 2 --force-mock
```

---

## Usage

All commands are dispatched through `main.py`.

### Train SAPG

```bash
python main.py train \
    --task allegrokuka --task-name regrasping \
    --num-envs 24576 --num-policies 6 \
    --seed 0 --output-dir results/regrasping
```

Ablation switches:

```bash
# Symmetric aggregation (all policies use all others' data)
python main.py train --task shadowhand --symmetric

# Disable off-policy aggregation entirely (independent PPO per block)
python main.py train --task shadowhand --no-off-policy

# High off-policy ratio (no subsampling of follower data)
python main.py train --task shadowhand --no-subsample

# Follower entropy coefficient
python main.py train --task reorientation --entropy-coef 0.005
```

### Train baselines

```bash
python main.py baseline --algo ppo --task allegrohand --seed 0
python main.py baseline --algo pbt --task regrasping --population-size 6
python main.py baseline --algo pql --task throw
```

### Ablations (Figure 6)

```bash
python main.py ablate --ablation entropy --task reorientation
python main.py ablate --ablation symmetric --task shadowhand
python main.py ablate --ablation no_off_policy --task shadowhand
python main.py ablate --ablation high_off_policy_ratio --task shadowhand
```

### Diversity analysis (Figures 7–8)

```bash
python main.py diversity --task allegrohand --num-transitions 400000
```

### Plot results (Figures 5–6, Table 1)

```bash
python main.py plot --results-dir results --figures-dir figures
```

### Evaluate a checkpoint

```bash
python main.py eval --checkpoint results/regrasping/sapg_regrasping_seed0.pt \
    --task allegrokuka --policy-index 0 --episodes 32
```

---

## Algorithm summary

### On-policy PPO loss (Eq. 2)

```
L_on(pi_i) = E[ min(r_t, clip(r_t, 1-eps, 1+eps)) * A_t ],   r_t = pi_theta / pi_old
```

### Off-policy importance-sampled loss (Eq. 3)

```
L_off(pi_i; X) = (1/|X|) sum_{j in X} E_{(s,a)~pi_j}[
    min(r_pi_i, clip(r_pi_i, mu(1-eps), mu(1+eps))) * A^{pi_i,old} ]
```
where `r_pi_i(s,a) = pi_i(s,a) / pi_j(s,a)` and `mu = pi_i,old(s,a) / pi_j(s,a)`.

### Combined objective (Eq. 4)

```
L(pi_i) = L_on(pi_i) + lambda * L_off(pi_i; X),   lambda = 1
```

### Critic targets

- On-policy, n-step (Eq. 5, `n=3`):
  `V_on_target(s_t) = sum_{k=t}^{t+2} gamma^{k-t} r_k + gamma^3 V_old(s_{t+3})`
- Off-policy, 1-step (Eq. 6):
  `V_off_target(s'_t) = r_t + gamma V_old(s'_{t+1})`

### Leader–follower aggregation (Sec. 4.3, 4.6)

- **Leader** (`i=1`): `X = {2..M}`; uses on-policy `D_1` + subsampled off-policy `D_1'`.
- **Followers** (`j=2..M`): `X = {}`; on-policy only.
- Off-policy batch size is subsampled to **match** the on-policy batch size.

### Follower entropy regularization (Sec. 4.5)

```
L(pi_i) = L_on(pi_i) + sigma * (i-1) * H(pi_i(a|s))
```
The leader receives no entropy bonus. `sigma in {0, 0.003, 0.005}`.

---

## Hyperparameters

| Parameter | AllegroKuka | ShadowHand | AllegroHand |
|---|---|---|---|
| `num_envs` (N) | 24576 | 24576 | 24576 |
| `num_policies` (M) | 6 | 6 | 6 |
| `horizon` (H) | 16 | 8 | 8 |
| `learning_rate` | 1e-4 | 5e-4 | 5e-4 |
| `clip_eps` | 0.1 | 0.1 | 0.2 |
| `mini_epochs` | 2 | 5 | 5 |
| `phi_dim` | 32 | 16 | 16 |
| `use_lstm` | yes (768) | no | no |
| actor hidden dims | 768-512-256 | 512-512-256-128 | 512-256-128 |

Shared: `gamma=0.99`, `tau=0.95`, `n_step=3`, `critic_coef=4.0`, `lambda_off=1.0`,
`bounds_coef=1e-4`, `max_grad_norm=1.0`, KL-adaptive LR threshold `0.016`.

---

## Expected results (Table 1, after 2e10 samples)

| Algorithm | AllegroHand | ShadowHand | Regrasping | Throw | Reorientation |
|---|---|---|---|---|---|
| PPO | 1.01e4 | 1.07e4 | 1.25 | 16.8 | 2.85 |
| PBT | 7.28e3 | 1.01e4 | 31.9 | 19.2 | 23.2 |
| PQL | 1.01e4 | 1.28e4 | 2.73 | 2.62 | 1.66 |
| SAPG (coef=0) | 1.23e4 | 1.17e4 | 35.7 | 23.7 | 33.2 |
| SAPG (coef=.005) | 9.14e3 | 1.28e4 | 33.4 | 18.7 | 38.6 |

---

## Compute notes

- Full runs: ~2e10 transitions, ~48–60 hrs per run on a single GPU.
- 5 seeds × 5 tasks × 4 methods is a heavy compute budget.
- For quick validation, use `--num-envs 1024 --num-policies 2 --force-mock`.

---

## Sanity checks

The implementation includes unit-testable invariants:

- `i=j` off-policy loss reduces to the on-policy loss (Eq. 3 reduction).
- n-step return (`n=3`) matches a manual rollout computation.
- `phi_j` receives gradients only from its own policy's objective.
- Off-policy subsample size equals the on-policy batch size.

---

## Citation

```
@inproceedings{sapg,
  title={SAPG: Split and Aggregate Policy Gradients},
  author={...},
  year={2024}
}
```
