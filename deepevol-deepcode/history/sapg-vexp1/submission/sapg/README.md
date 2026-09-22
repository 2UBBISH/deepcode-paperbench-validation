# SAPG: Split and Aggregate Policy Gradients

A reproduction of **"SAPG: Split and Aggregate Policy Gradients"** — an on-policy RL
algorithm that scales to tens of thousands of parallel environments by:

1. **Splitting** the environment pool into `M` blocks, each trained by a *diverse*
   leader/follower policy that shares a common backbone `B_theta` / `C_psi` but is
   conditioned on a per-policy latent `phi_j`.
2. **Aggregating** the data of all policies into the leader's update via an
   importance-sampled **off-policy** objective, which overcomes PPO's batch-size
   saturation.

---

## 1. Repository layout

```
sapg/
├── main.py                      # Entry point: train/eval dispatch
├── train.py                     # Training driver (SAPG/PPO/DexPBT/PQL)
├── eval.py                      # Evaluation + success/reward metrics
├── configs/
│   ├── allegrokuka.yaml         # phi_dim=32, horizon=16, clip=0.1, LSTM
│   ├── shadowhand.yaml          # phi_dim=16, horizon=8,  clip=0.1
│   ├── allegrohand.yaml         # phi_dim=16, horizon=8,  clip=0.2
│   └── sapg.yaml                # M=6, N=24576, lambda=1, sigma grid
├── sapg/
│   ├── actor_critic.py          # Shared backbone + phi_j conditioning
│   ├── networks.py              # MLP/LSTM builders (ELU), Gaussian head
│   ├── losses.py                # L_on (Eq.2), L_off (Eq.3), critic (Eq.5-9), entropy (Eq.10)
│   ├── rollout.py               # Per-policy buffers, n-step returns (n=3)
│   ├── aggregation.py           # Leader-follower / symmetric schemes, subsampling
│   └── sapg_trainer.py          # Algorithm 1 orchestration
├── envs/
│   ├── isaacgym_wrapper.py      # Vectorized env wrapper (24576 parallel)
│   ├── allegrokuka_tasks.py     # Regrasping, Throw, Reorientation + curriculum
│   ├── shadowhand_task.py       # 24-DoF in-hand reorientation
│   └── allegrohand_task.py      # 16-DoF in-hand reorientation
├── baselines/
│   ├── ppo.py                   # Vanilla PPO (scaled batch)
│   ├── dexpbt.py                # Population-based training, M=6, mutation
│   └── pql.py                   # Parallel Q-learning (DDPG + mixed exploration)
├── diversity/
│   ├── pca_metric.py            # PCA reconstruction error vs k (Fig 7)
│   └── mlp_metric.py            # 2-layer MLP reconstruction error (Fig 8)
├── utils/
│   ├── logger.py                # Mean/std-error logging, curve plotting
│   ├── kl_lr.py                 # KL-adaptive LR (threshold 0.016)
│   └── checkpoint.py
└── scripts/
    ├── run_all.sh               # 5 tasks x 4 methods x 5 seeds
    └── run_ablations.sh         # symmetric, high-off-policy, entropy coefs
```

---

## 2. Installation

```bash
# 1. Core Python dependencies
pip install -r requirements.txt

# 2. NVIDIA IsaacGym (NOT available on PyPI — install manually)
#    Download "Isaac Gym Preview 4" from https://developer.nvidia.com/isaac-gym
cd isaacgym/python && pip install -e . && cd -

# 3. Verify
python -c "import isaacgym; print('IsaacGym OK')"
```

**Hardware:** a single NVIDIA GPU (A100 / RTX-class) with enough VRAM for
24 576 parallel environments + an LSTM policy. A full run is ~48–60 h and
~2 × 10¹⁰ transitions.

> **No GPU?** The codebase ships a NumPy `DummyVectorEnv` fallback. Set
> `SAPG_FORCE_DUMMY_ENV=1` (or simply run without IsaacGym) to smoke-test the
> full pipeline on CPU with small `--num-envs`.

---

## 3. Quick start

```bash
# SAPG on AllegroKuka Regrasping (default: 24576 envs, M=6, 5 seeds via script)
python main.py --method sapg --task allegrokuka_regrasping \
               --config configs/allegrokuka.yaml

# Small CPU smoke test
SAPG_FORCE_DUMMY_ENV=1 python main.py --method sapg --task allegrohand \
               --config configs/allegrohand.yaml --num-envs 64

# Baselines
python main.py --method ppo    --task allegrokuka_regrasping --config configs/allegrokuka.yaml
python main.py --method dexpbt --task allegrokuka_regrasping --config configs/allegrokuka.yaml
python main.py --method pql    --task allegrohand          --config configs/allegrohand.yaml

# Evaluation
python main.py --eval --checkpoint runs/sapg/checkpoint_latest.pt \
               --task allegrokuka_regrasping --eval-episodes 100
```

### Reproducing the paper

```bash
bash scripts/run_all.sh          # Table 1 + Figure 5 (5 tasks x 4 methods x 5 seeds)
bash scripts/run_ablations.sh    # Figure 6 (symmetric / high-off-policy / entropy grid)
```

---

## 4. Algorithm summary (Algorithm 1)

For each iteration:

1. **Collect** — for every policy `j ∈ {1..M}`, roll out its env block with horizon
   `H` (16 for AllegroKuka, 8 for ShadowHand/AllegroHand) into buffer `D_j`.
2. **Subsample** — draw `|D_1|` transitions uniformly from `∪_{j≥2} D_j` to form
   `D_1'` (equal on-/off-policy volume).
3. **Update** — minimize

   ```
   L = L_off(D_1'; {2..M}) + L_on(D_1) + Σ_{j≥2} L_on(D_j) [+ entropy]
   ```

   * `L_on` — PPO clipped surrogate (Eq. 2), `eps = 0.1` (0.2 AllegroHand).
   * `L_off` — importance-sampled surrogate (Eq. 3) with behaviour ratio
     `r = pi_i(s,a)/pi_j(s,a)` clipped to `mu(1±eps)`, `mu = pi_{i,old}/pi_j`.
   * `L_critic = L_critic_on + λ·L_critic_off`, `λ = 1`, critic coef `4.0`.
   * Entropy (Eq. 10) applied **only to followers**: `σ·(i-1)·H(pi)`.
4. **Optimize** — Adam (PyTorch defaults), grad-norm clip `1.0`,
   KL-adaptive LR (threshold `0.016`), mini-batch `num_envs·4`,
   mini-epochs `2` (AllegroKuka) / `5` (easy tasks).

`phi_j` is updated **only** from policy `j`'s own objective; `theta`/`psi` are
shared and updated from all objectives.

---

## 5. Hyperparameters

| Setting | AllegroKuka | ShadowHand | AllegroHand |
|---|---|---|---|
| DoF | 23 | 24 | 16 |
| `phi_dim` | 32 | 16 | 16 |
| Actor | MLP[768,512,256] ELU → LSTM(768) | MLP[512,512,256,128] ELU | MLP[512,256,128] ELU |
| `horizon` | 16 | 8 | 8 |
| `clip_eps` | 0.1 | 0.1 | 0.2 |
| `mini_epochs` | 2 | 5 | 5 |
| `entropy_coef` | 0.0 (0.005 Reorientation) | 0.0 | 0.0 |
| Metric | successes | episode reward | episode reward |

Shared across tasks: `gamma=0.99`, `tau=0.95`, `n_step=3`, `lam=1.0`,
`critic_coef=4.0`, `lr=3e-4`, `kl_threshold=0.016`, `num_policies=6`,
`num_envs=24576`, `target_transitions=2e10`.

---

## 6. Documented defaults (paper ambiguities)

The paper leaves several details unspecified; the following defaults are used
and are configurable:

| Ambiguity | Chosen default |
|---|---|
| `phi_j` injection method | concatenation to network input |
| Critic layer dims | mirror the actor backbone |
| Reward weights | DexPBT reference values (`reach=1, lift=2, target=5, success=10`) |
| Adam betas/eps | PyTorch defaults (`0.9, 0.999, 1e-8`) |
| KL-adaptive LR bounds | `[1e-5, 1e-2]`, `×1.5` up / `×0.5` down |
| Off-policy weight normalization | rely on `mu(1±eps)` clipping |
| Latent init | zeros (early training ≈ single policy) |
| Sigma init | `log_std = -1.0` per action dim |

---

## 7. Expected results (Table 1, after ~2e10 transitions, 5 seeds)

**Hard tasks — successes (mean ± s.e.):**

| Method | Regrasping | Throw | Reorientation |
|---|---|---|---|
| PPO | ~1.25 | ~0 | ~0 |
| PQL | ~2.73 | ~0 | ~0 |
| DexPBT | 31.9 | 19.2 | 23.2 |
| **SAPG (σ=0)** | **35.7 ± 1.46** | **23.7 ± 0.74** | 33.2 ± 4.20 |
| **SAPG (σ=0.005)** | — | — | **38.6 ± 0.63** |

**Easy tasks — episode reward:**

| Method | AllegroHand | ShadowHand |
|---|---|---|
| PQL | 1.01e4 | 1.28e4 |
| **SAPG (σ=0)** | **1.23e4** | 1.17e4 |

**Ablations (Fig. 6):** symmetric aggregation and high off-policy ratio both
degrade performance; removing the off-policy combination is significantly
worse; `σ=0.005` improves Reorientation by ~16.5 %.

**Diversity (Figs. 7–8):** SAPG exhibits slower PCA reconstruction-error decay
vs `k` and consistently higher MLP reconstruction error than PPO — i.e. SAPG's
policies are more behaviourally diverse.

---

## 8. Diversity metrics

```bash
# PCA reconstruction error vs number of components (Figure 7)
python -c "
from sapg.diversity.pca_metric import pca_diversity_curve, plot_pca_diversity
curves = pca_diversity_curve({'sapg': data_sapg, 'ppo': data_ppo})
plot_pca_diversity(curves, filename='fig7.png')
"

# 2-layer MLP reconstruction error vs hidden size (Figure 8)
python -c "
from sapg.diversity.mlp_metric import mlp_diversity_curve, plot_mlp_diversity
curves = mlp_diversity_curve({'sapg': data_sapg, 'ppo': data_ppo})
plot_mlp_diversity(curves, filename='fig8.png')
"
```

Both metrics pool 400 k `(state, action)` transitions per policy and report
reconstruction error (PCA: normalized MSE vs `k`; MLP: validation MSE vs hidden
width, ReLU, Adam defaults, L2 loss).

---

## 9. Notes

* All losses return **negative objectives** (minimization convention).
* The leader is index `0` in code (paper's `i = 1`); followers are `1..M-1`.
* `num_envs` must be divisible by `num_policies`.
* Checkpoints store model, optimizer, KL-LR scheduler, iteration/transition
  counters and RNG state for exact resumption.
