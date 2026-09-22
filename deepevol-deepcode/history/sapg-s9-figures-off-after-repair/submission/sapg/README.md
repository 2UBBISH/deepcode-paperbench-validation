# SAPG: Split and Aggregate Policy Gradients

Reference implementation of **SAPG** — an on-policy reinforcement learning algorithm
that scales to tens of thousands of parallel environments by *splitting* them into
`M` blocks trained by diverse leader/follower policies, then *aggregating* follower
data into a single leader update via importance-sampled off-policy PPO updates.

The shared actor backbone `B_theta` and critic backbone `C_psi` are conditioned on
per-policy latent vectors `phi_j` (the *same* `phi_j` for the actor and critic of
policy `j`). `theta` and `psi` receive gradients from **all** objectives, while each
`phi_j` is updated **only** by policy `j`'s own objective.

---

## 1. Repository layout

```
sapg/
├── main.py                      # CLI entry point (task/algorithm selection, launch training)
├── sapg/
│   ├── __init__.py
│   ├── algorithm.py             # SAPG training loop (Algorithm 1)
│   ├── losses.py                # L_on (Eq.2), L_off (Eq.3), combined (Eq.4), critic losses (Eq.7-9)
│   ├── networks.py              # Shared actor B_theta / critic C_psi + phi_j conditioning
│   ├── rollout.py               # Per-policy data collection, env-block assignment, buffers D_1..D_M
│   ├── returns.py               # GAE, 3-step on-policy targets (Eq.5), 1-step off-policy targets (Eq.6)
│   ├── aggregation.py           # leader_follower / symmetric / high_off_policy_ratio / no_off_policy
│   ├── entropy.py               # Follower entropy regularization sigma*(i-1)*H; per-block learnable sigma
│   └── config.py                # Hyperparameter dataclasses per task (Tables 2-4)
├── envs/
│   ├── isaacgym_wrapper.py      # IsaacGym vectorized env wrapper (N=24576 envs, block splitting)
│   ├── allegrokuka.py           # Regrasping / Throw / Reorientation (23-DoF, hard tasks)
│   ├── shadowhand.py            # 24-DoF in-hand reorientation (easy task)
│   └── allegrohand.py           # 16-DoF in-hand reorientation (easy task)
├── baselines/
│   ├── ppo.py                   # Vanilla PPO with scaled batch size (Fig. 2)
│   ├── pql.py                   # Parallel Q-Learning (Li et al. 2023)
│   └── dexpbt.py                # Population-based training + PPO (Petrenko et al. 2023)
├── experiments/
│   ├── train.py                 # Train SAPG/baselines, 5 seeds, log curves
│   ├── ablations.py             # symmetric / high-off-policy / no-off-policy / entropy coefs
│   ├── diversity.py             # PCA (Fig.7) & MLP (Fig.8) reconstruction-error metrics
│   └── plot.py                  # Reproduce Fig.2, Fig.5, Fig.6, Table 1
├── README.md
└── requirements.txt
```

---

## 2. Installation

```bash
# 1. Core Python dependencies
pip install -r requirements.txt

# 2. IsaacGym (optional but required for real physics / N=24576 envs)
#    Download from https://developer.nvidia.com/isaac-gym and then:
cd isaacgym/python && pip install -e . && cd -

# 3. (Optional) IsaacGymEnvs task definitions used by the wrappers
#    https://github.com/NVIDIA-Omniverse/IsaacGymEnvs
```

> **No IsaacGym?** The codebase degrades gracefully: `envs/isaacgym_wrapper.py`
> exposes a pure-PyTorch `DummyVectorEnv` fallback so the full SAPG loop can be
> smoke-tested on CPU (`--dry-run`).

---

## 3. Quick start

```bash
cd sapg

# List available tasks and algorithms
python main.py --list-tasks

# Print the resolved config for a task (no training)
python main.py --task regrasping --algorithm sapg --print-config

# Show the paper's expected Table 1 numbers
python main.py --expected-results

# Smoke test the full SAPG loop on CPU (dummy env, few iterations)
python main.py --task regrasping --algorithm sapg --dry-run --num-iterations 5

# Real training run (requires IsaacGym + GPU)
python main.py --task regrasping --algorithm sapg --seed 0
```

### CLI highlights

| Flag | Meaning |
| --- | --- |
| `--task {regrasping,throw,reorientation,shadowhand,allegrohand}` | Task selection |
| `--algorithm {sapg,ppo,pql,dexpbt}` | Algorithm selection |
| `--num-envs N` | Number of parallel envs (default 24576) |
| `--num-blocks M` | Number of SAPG blocks (default 6) |
| `--aggregation {leader_follower,symmetric,high_off_policy_ratio,no_off_policy}` | Aggregation variant |
| `--entropy-coef SIGMA` | Follower entropy coefficient `sigma` |
| `--no-subsample-off-policy` | Leader uses **all** off-policy data (ablation) |
| `--no-adaptive-lr` | Disable KL-based adaptive LR |
| `--dry-run` | Use `DummyVectorEnv` (CPU smoke test) |
| `--seed S` | Random seed |

---

## 4. Algorithm summary (Algorithm 1)

1. Initialise `theta`, `psi`, `phi_1..phi_M`, `N` envs, buffers `D_1..D_M`.
2. For each outer iteration, for `j = 1..M`: collect `D_j` from env block `j`
   using `B_theta` conditioned on `phi_j`.
3. Sample `|D_1|` transitions from `union_{j=2}^{M} D_j` → `D_1'` (off-policy subsampling).
4. Build the total loss

   ```
   L = OffPolicyLoss(D_1') + OnPolicyLoss(D_1) + sum_{j=2}^{M} OnPolicyLoss(D_j)
       [+ entropy terms]
   ```

5. Update `theta`, `psi` by gradient descent; update each `phi_j` with its own objective.

### Losses

* **On-policy PPO (Eq. 2)**
  `L_on(pi_theta) = E_{pi_old}[ min( r_t, clip(r_t, 1-eps, 1+eps) ) * A_t^{pi_old} ]`
  with `r_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t)`.

* **Off-policy importance-sampled (Eq. 3)**
  `L_off(pi_i; X) = (1/|X|) sum_{j in X} E_{(s,a)~pi_j}[ min( r_{pi_i}, clip(r_{pi_i}, mu(1-eps), mu(1+eps)) ) * A^{pi_{i,old}} ]`
  where `r_{pi_i} = pi_i(s,a) / pi_j(s,a)` and `mu = pi_{i,old}(s,a) / pi_j(s,a)`.
  When `i = j`, `pi_j = pi_{i,old}` so `mu = 1` and Eq. 3 reduces to Eq. 2.

* **Combined (Eq. 4)** `L(pi_i) = L_on(pi_i) + lambda * L_off(pi_i; X)`, `lambda = 1`.

* **Critic targets**
  * On-policy 3-step (Eq. 5): `V_on^target(s_t) = sum_{k=t}^{t+2} gamma^{k-t} r_k + gamma^3 V_{pi_j,old}(s_{t+3})`
  * Off-policy 1-step (Eq. 6): `V_off^target(s_t') = r_t + gamma * V_{pi_j,old}(s_{t+1}')`
  * `L^critic = L_on^critic + lambda * L_off^critic` (Eq. 9), scaled by `lambda' = 4.0`.

* **Entropy (Sec 4.5)** Follower `i` adds `sigma * (i-1) * H(pi(a|s))`; the leader has **no** entropy term.

---

## 5. Hyperparameters (Tables 2-4)

| Setting | AllegroKuka (hard) | ShadowHand | AllegroHand |
| --- | --- | --- | --- |
| Envs `N` | 24576 | 24576 | 24576 |
| Blocks `M` | 6 | 6 | 6 |
| Horizon | 16 | 8 | 8 |
| Latent dim `phi_j` | 32 | 16 | 16 |
| Actor | MLP[768,512,256] ELU → LSTM(768) | MLP[512,512,256,128] ELU | MLP[512,256,128] ELU |
| Clip `eps` | 0.1 | 0.1 | 0.2 |
| LR | 1e-4 | 5e-4 | 5e-4 |
| Mini-epochs | 2 | 5 | 5 |
| Entropy `sigma` | 0 (0.005 for Reorientation) | 0 | 0 |

Shared across tasks: `gamma = 0.99`, GAE `tau = 0.95`, grad-norm clip `1.0`,
KL threshold `0.016` (adaptive LR), Adam optimizer, ELU activations.

---

## 6. Reproducing the paper

### 6.1 Main results (Table 1, Fig. 5)

```bash
cd sapg
python -m experiments.train \
    --tasks regrasping throw reorientation shadowhand allegrohand \
    --algorithms sapg ppo pql dexpbt \
    --seeds 0 1 2 3 4 \
    --total-transitions 2e10 \
    --output-dir runs
```

Each run is ~48-60 hrs on a single GPU and collects ~2e10 transitions.
Results are written to `runs/summary.json` and `runs/curves.npz`.

### 6.2 PPO saturation (Fig. 2)

```bash
python -m experiments.train --suite ppo_saturation --output-dir runs
```

Trains vanilla PPO with batch sizes scaled from 128 to 24576 envs and confirms
asymptotic performance saturates beyond a certain batch size.

### 6.3 Ablations (Fig. 6)

```bash
python -m experiments.ablations --suite all --output-dir runs_ablations
```

Runs the aggregation variants (`leader_follower`, `symmetric`,
`high_off_policy_ratio`, `no_off_policy`) and the entropy sweep
`sigma in {0, 0.003, 0.005}`.

### 6.4 Diversity metrics (Fig. 7 & Fig. 8)

```bash
# Dump state-transitions during training, then:
python -m experiments.diversity --runs-dir runs --output-dir diversity
```

* **PCA (Fig. 7):** reconstruction error using the top-`k` PCA components;
  SAPG should show the *slowest* decrease.
* **MLP (Fig. 8):** 2-layer MLP auto-encoder (ReLU, Adam defaults, L2 loss,
  400k state-transitions); SAPG should show consistently *higher* training error
  than PPO.

### 6.5 Figures and table

```bash
python -m experiments.plot --runs-dir runs --output-dir figures
```

Produces Fig. 2, Fig. 5, Fig. 6 and a text/LaTeX rendering of Table 1
(compared against the paper's reported values when available).

---

## 7. Expected results (Table 1, after 2e10 samples, 5 seeds, mean ± std error)

| Task | SAPG (sigma=0) | SAPG (sigma=0.005) | PPO | DexPBT | PQL |
| --- | --- | --- | --- | --- | --- |
| AllegroHand (reward) | 1.23e4 ± 3.29e2 | 9.14e3 ± 8.38e2 | — | — | — |
| ShadowHand (reward) | 1.17e4 ± 2.64e2 | 1.28e4 ± 2.80e2 | — | — | — |
| Regrasping (succ/ep) | 35.7 ± 1.46 | 33.4 ± 2.25 | 1.25 ± 1.15 | 31.9 ± 2.26 | 2.73 ± 0.02 |
| Throw (succ/ep) | 23.7 ± 0.74 | 18.7 ± 0.43 | 16.8 ± 0.48 | 19.2 ± 1.07 | 2.62 ± 0.08 |
| Reorientation (succ/ep) | 33.2 ± 4.20 | 38.6 ± 0.63 | 2.85 ± 0.05 | 23.2 ± 4.86 | 1.66 ± 0.11 |

**Success criteria:** SAPG beats DexPBT by 12-66% on hard tasks; beats PQL by ~21%
on AllegroHand; is comparable on ShadowHand; PPO/PQL fail on hard tasks.

---

## 8. Integrity tests

The following invariants are checked by the test suite / assertions:

1. **Eq. 3 → Eq. 2 consistency:** when `i = j`, `behavior_log_probs == old_log_probs`,
   so `mu = 1` and the off-policy loss reduces exactly to the on-policy loss.
2. **Subsampling size:** `|D_1'| = |D_1|` after off-policy subsampling.
3. **Latent gradient isolation:** `phi_j` receives gradients only from policy `j`'s
   objective (`networks.zero_other_latent_grads(j)`).
4. **Leader entropy:** the leader (`i = 1`) has no entropy term.

Run a quick check:

```bash
python -c "
import torch
from sapg.losses import on_policy_loss, off_policy_loss
lp = torch.randn(64); adv = torch.randn(64)
on = on_policy_loss(lp, lp, adv, clip_eps=0.1).loss
off = off_policy_loss(lp, lp, lp, adv, clip_eps=0.1).loss
assert torch.allclose(on, off, atol=1e-6), (on, off)
print('Eq.3 -> Eq.2 consistency OK')
"
```

---

## 9. Notes on unspecified details

Where the paper leaves details unspecified, the following conventions are used:

* **`phi_j` conditioning:** concatenated to the observation/state input.
* **Critic layer dims:** mirror the actor backbone dims.
* **GAE lambda:** `0.95` (the paper's `tau`).
* **Reward weights:** follow Petrenko et al. 2023 (AllegroKuka) and
  Li et al. 2023 (hands).
* **Optimizer:** Adam with per-task learning rates (see Table above).

---

## 10. Citation

```bibtex
@inproceedings{sapg2024,
  title     = {SAPG: Split and Aggregate Policy Gradients},
  author    = {Anonymous},
  booktitle = {International Conference on Machine Learning},
  year      = {2024}
}
```
