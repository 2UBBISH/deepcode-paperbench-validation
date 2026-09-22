# SAPG: Split and Aggregate Policy Gradients

Reproduction of **"SAPG: Split and Aggregate Policy Gradients"**.

SAPG is an on-policy RL algorithm for massively parallel simulators. The `N`
environments are split into `M` contiguous **blocks**; each block is trained by
its own policy (`1 leader + M-1 followers`) that shares an actor backbone
`B_theta` and a critic backbone `C_psi`, conditioned on a per-policy latent
`phi_j`. All blocks' data is fused into the **leader** through an
importance-sampled off-policy PPO update:

```
L = L_on(D_1) + lambda * L_off(D'_1) + sum_{j>=2} L_on(D_j) + lambda' * L_critic   (Eq. 2-9)
```

This yields higher asymptotic performance than vanilla PPO at large batch sizes
(a saturation effect at `N >= ~10k` environments for a single policy).

---

## Installation

```bash
# 1) Python 3.8-3.10 recommended
pip install -r requirements.txt

# 2) IsaacGym (GPU parallel simulator) - NOT on PyPI, manual install required:
#    download from https://developer.nvidia.com/isaac-gym and
#    cd isaacgym/python && pip install -e .
```

Optional dependencies degrade gracefully:

| Feature | Package | Fallback |
|---|---|---|
| Parallel simulation (`N=24576`) | `isaacgym`, `isaacgymenvs` | pure-PyTorch surrogate env (slower, same semantics) |
| PCA diversity metric (Fig. 7) | `scikit-learn` | NumPy SVD |
| MLP diversity metric (Fig. 8) | `torch` | NumPy random-feature ridge fallback |
| Figures | `matplotlib` | analysis still writes JSON |
| TensorBoard / W&B logging | `tensorboard`, `wandb` | console + JSONL logging |

Force the dependency-free surrogate backend at any time with:

```bash
export SAPG_FORCE_SURROGATE=1
```

---

## Repository layout

```
sapg/
├── sapg/
│   ├── algorithms/
│   │   ├── sapg.py        # Algorithm 1: leader/follower train loop, summed loss -> one backward
│   │   ├── ppo.py         # PPO base trainer (on-policy), KL-adaptive LR, bounds loss
│   │   └── rollout.py     # BlockManager + RolloutCollector (per-block data D_1..D_M)
│   ├── losses/
│   │   ├── ppo_loss.py        # L_on (Eq. 2) + entropy term (Eq. 10) + bounds loss
│   │   ├── off_policy_loss.py # L_off with mu-corrected IS ratio (Eq. 3) + combine (Eq. 4)
│   │   └── critic_loss.py     # 3-step / 1-step targets and value losses (Eqs. 5-9)
│   ├── models/
│   │   ├── networks.py    # MLP / LSTM / RecurrentBackbone / learnable sigma
│   │   ├── actor.py       # shared B_theta conditioned on phi_j (+ tanh-squashed Gaussian)
│   │   └── critic.py      # shared C_psi conditioned on phi_j
│   ├── aggregation/
│   │   ├── leader_follower.py # §4.3 controller + uniform subsampling |D'_1| = |D_1|
│   │   └── symmetric.py       # §4.2 symmetric ablation (+ high off-policy ratio mode)
│   ├── envs/
│   │   ├── isaac_env.py   # IsaacGym / surrogate vectorised env facade (N=24576)
│   │   ├── allegro_kuka.py# Regrasping / Throw / Reorientation
│   │   ├── shadow_hand.py # Shadow Hand in-hand reorientation
│   │   ├── allegro_hand.py# Allegro Hand in-hand reorientation
│   │   └── curriculum.py  # success tolerance 7.5cm -> 1cm
│   ├── buffers/rollout_buffer.py  # per-policy D_j, GAE, n-step/1-step targets, mu, minibatches
│   ├── baselines/
│   │   ├── ppo_baseline.py# vanilla PPO baseline + saturation sweep (Fig. 2 concept)
│   │   ├── pql.py         # parallel Q-learning (parallel DDPG w/ mixed exploration noise)
│   │   └── dexpbt.py      # PPO + population-based training (M=6 groups)
│   ├── analysis/
│   │   ├── diversity_pca.py # §6.4 Fig. 7: PCA reconstruction error vs k
│   │   └── diversity_mlp.py # §6.4 Fig. 8: MLP reconstruction error vs width
│   └── utils/             # config, GAE, returns, logging
├── configs/
│   ├── allegro_kuka.yaml  # Table 2 (Regrasping / Throw / Reorientation)
│   ├── shadow_hand.yaml   # Table 3
│   ├── allegro_hand.yaml  # Table 4
│   └── ablations.yaml     # Fig. 6 variants + lambda / num_policies sweeps
├── scripts/
│   ├── train.py              # SAPG / PPO / PQL / DexPBT entry point
│   ├── train_baseline.py     # baselines only, multi-seed + paper band
│   ├── evaluate.py           # checkpoint evaluation, Table-1 style summary
│   ├── run_ablation.py       # Figure 6 ablation study
│   └── run_diversity_analysis.py  # Figures 7-8
└── tests/                 # unit tests for losses / ratio / critic targets
```

---

## Quick start

```bash
# Smoke-test everything on CPU with the surrogate environment (no IsaacGym needed)
cd sapg
SAPG_FORCE_SURROGATE=1 python scripts/train.py --method sapg --task regrasping \
    --num-envs 256 --iterations 5 --verbose

# Real run: SAPG on Regrasping (Appendix B.1 / Table 2 config)
python scripts/train.py --config configs/allegro_kuka.yaml --task regrasping \
    --log-dir runs/regrasping --tensorboard

# Vanilla PPO baseline on all 24576 envs
python scripts/train.py --method ppo --task regrasping --num-envs 24576

# Baselines with 5-seed aggregation + paper shaded band
python scripts/train_baseline.py --method dexpbt --task reorientation --num-seeds 5
```

### Command-line flags (train.py)

| Flag | Meaning |
|---|---|
| `--task` | `regrasping`, `throw`, `reorientation`, `shadow_hand`, `allegro_hand` |
| `--method` | `sapg`, `ppo`, `pql`, `dexpbt` |
| `--aggregation` | `leader_follower` (default), `symmetric`, `none` |
| `--num-envs` / `--num-policies` | `N` (default 24576) and `M` (default 6) |
| `--entropy-coefficient` | follower entropy bonus `sigma` in Eq. 10 (`0`, `0.003`, `0.005`) |
| `--off-policy-weight` | `lambda` in Eq. 4 (default `1.0`) |
| `--no-subsample` | "high off-policy ratio" ablation (`\|D'_1\| > \|D_1\|`) |
| `--random-phi` | freeze `phi_j` at its random initialisation (diversity baseline) |
| `--surrogate` | force the pure-PyTorch surrogate env |
| `--max-samples` | stop after this many environment transitions |

---

## Algorithm details and where they live

### 1. Split-and-aggregate loop (Algorithm 1, §4.6)
`SAPGTrainer.update()` performs one outer iteration:

1. `RolloutCollector.collect()` rolls each block `j` with its own policy for
   `horizon_length` steps, producing buffers `D_1..D_M`.
2. GAE advantages (`tau = 0.95` interpreted as the GAE lambda) and value targets
   are computed per buffer (`sapg/utils/gae.py`, `sapg/utils/returns.py`).
3. The leader's off-policy batch `D'_1` is built by flattening the follower
   buffers and **uniformly subsampling to `|D_1|` transitions** (§4.3), so the
   off-policy term has the same weight as the on-policy term.
4. The summed objective is minimised with a single backward pass:

   ```
   L = sum_j L_on(D_j)  +  lambda * L_off(D'_1)  +  lambda' * L_critic
   ```

5. `phi_j` is updated **only** from policy `j`'s own objective (§4.4); the
   backbones `B_theta` / `C_psi` receive gradients from all objectives.

### 2. On-policy loss (Eq. 2, Eq. 10)
`sapg/losses/ppo_loss.py`: clipped surrogate with
`r_t = exp(log pi_theta - log pi_old)` and `clip(r_t, 1-eps, 1+eps)`.
Followers add `sigma * (j-1) * H(pi(a|s))`; the leader is excluded.

### 3. Off-policy loss (Eq. 3, Eq. 4)
`sapg/losses/off_policy_loss.py`: for leader `i=1` and source policy `j`,

```
r_pi_i(s,a) = pi_i(s,a) / pi_j(s,a),      mu = pi_{i,old}(s,a) / pi_j(s,a)
L_off = E_{j in X} min(r_pi_i * A,  clamp(r_pi_i, mu(1-eps), mu(1+eps)) * A),
X = {2..M},  lambda = 1
```

When `i == j` (i.e. `r = mu = 1`) this reduces exactly to the on-policy
surrogate — asserted by `tests/test_importance_ratio.py`.

### 4. Critic targets and loss (Eqs. 5-9)
`sapg/losses/critic_loss.py` + `sapg/utils/returns.py`:

* 3-step **on-policy** target: `V_on(s_t) = sum_{k=t}^{t+2} gamma^{k-t} r_k + gamma^3 V_old(s_{t+3})`
* 1-step **off-policy** target: `V_off(s'_t) = r_t + gamma * V_old(s'_{t+1})`
* `L_critic = lambda' * (L_on + lambda * L_off)`, with `lambda' = 4.0`

### 5. Aggregation schemes
- `leader_follower` (§4.3, default): only policy 1 receives off-policy data.
- `symmetric` (§4.2, ablation): every policy receives all other policies' data.
- `none`: pure multi-policy PPO (no aggregation).

### 6. Curriculum (§5.1, Appendix A)
`sapg/envs/curriculum.py` starts with a 7.5 cm success tolerance and multiplies
it by 0.9 (down to a 1 cm floor) each time the average number of successes per
episode crosses 3, holding the criterion for `K = 30` consecutive steps.

---

## Hyperparameters (Appendix B, Tables 2-4)

| | AllegroKuka (Regrasping / Throw / Reorientation) | Shadow Hand | Allegro Hand |
|---|---|---|---|
| envs `N` | 24576 | 24576 | 24576 |
| policies `M` | 6 | 6 | 6 |
| learning rate | 1e-4 | 5e-4 | 5e-4 |
| clamp `eps` | 0.1 | 0.2 | 0.2 |
| horizon | 16 | 8 | 8 |
| mini-epochs | 2 | 5 | 5 |
| actor / critic | MLP 768-512-256 ELU + LSTM(1, 768) | MLP 512-512-256-128 ELU | MLP 512-256-128 ELU |
| `phi_j` dim | 32 | 16 | 16 |
| entropy `sigma` | 0.0 (0.005 for Reorientation) | 0.0 | 0.0 |

Shared defaults: `gamma = 0.99`, `tau = 0.95` (GAE lambda), grad-norm clip 1.0,
KL threshold 0.016, minibatch = `num_envs * 4`, critic coefficient `lambda' = 4.0`,
off-policy weight `lambda = 1.0`, bounds-loss coefficient `1e-4`, Adam
(`betas=(0.9, 0.999)`, `eps=1e-8`).

---

## Reproducing the paper's results

All experiments use 24576 environments and 5 seeds; curves are reported versus
collected environment transitions, with a shaded band computed exactly as in
§5.2: `band(t) = (2 / sqrt(n)) * sum_i (mean(t) - y_i(t))^2`.

### Table 1 / Figure 5 — main performance (2e10 samples)

```bash
# SAPG (sigma = 0)
for task in regrasping throw reorientation shadow_hand allegro_hand; do
    python scripts/train.py --method sapg --task $task --num-seeds 5 --max-samples 2e10
done

# SAPG (sigma = 0.005)
python scripts/train.py --method sapg --task reorientation --entropy-coefficient 0.005 \
    --num-seeds 5 --max-samples 2e10

# Baselines
python scripts/train_baseline.py --method ppo    --task regrasping --num-seeds 5
python scripts/train_baseline.py --method pql    --task allegro_hand --num-seeds 5
python scripts/train_baseline.py --method dexpbt --task throw --num-seeds 5
```

Reference final numbers (mean ± band):

| Task | SAPG (sigma=0) | SAPG (sigma=0.005) |
|---|---|---|
| Allegro Hand | 1.23e4 ± 3.29e2 | 9.14e3 ± 8.38e2 |
| Shadow Hand | 1.17e4 ± 2.64e2 | 1.28e4 ± 2.80e2 |
| Regrasping | 35.7 ± 1.46 | 33.4 ± 2.25 |
| Throw | 23.7 ± 0.74 | 18.7 ± 0.43 |
| Reorientation | 33.2 ± 4.20 | **38.6 ± 0.63** |

Expected qualitative findings:
* SAPG beats DexPBT by 12-66% on Regrasping / Throw / Reorientation (~66% better on Reorientation).
* SAPG beats PQL by ~21% on Allegro Hand and is comparable on Shadow Hand.
* PPO and PQL fail (near-zero success) on the hard AllegroKuka tasks.
* `sigma = 0` is best for Shadow Hand, Allegro Hand, Regrasping, Throw;
  `sigma = 0.005` is best for Reorientation (+16.5% over `sigma = 0`).

### Figure 2 concept — PPO saturation

```bash
python -c "
from sapg.baselines.ppo_baseline import ppo_saturation_sweep, saturation_summary
sweep = ppo_saturation_sweep(task='regrasping', seeds=(0,))
print(saturation_summary(sweep))
"
```

Confirms asymptotic performance saturates beyond ~10k environments.

### Figure 6 — ablations

```bash
python scripts/run_ablation.py --task regrasping --num-seeds 5
# fast smoke test without a GPU:
python scripts/run_ablation.py --synthetic --output-dir runs/fig6
```

Variants (`configs/ablations.yaml`): `sapg`, `entropy`, `high_off_policy_ratio`,
`symmetric`, `no_off_policy`, plus entropy sweeps over `sigma in {0, 0.003, 0.005}`.

Expected: symmetric aggregation is significantly worse everywhere; removing the
off-policy term hurts; removing subsampling ("high off-policy ratio") is worse on
Shadow Hand / Allegro Hand and marginally worse on Regrasping / Throw.

### Figures 7-8 — state-space diversity

```bash
# collect 400k transitions per method and fit both metrics
python scripts/run_diversity_analysis.py --task regrasping --train-samples 100000
# fast smoke test with synthetic state batches:
python scripts/run_diversity_analysis.py --synthetic --output-dir runs/diversity
```

* **Fig. 7 (PCA)**: reconstruction error of visited states versus the number of
  principal components `k`. SAPG shows the *slowest* error decrease with `k`
  (more state-space directions explored) versus PPO and a random policy.
* **Fig. 8 (MLP)**: two-hidden-layer ReLU auto-reconstructor of equal width `w`
  (Adam defaults, L2 reconstruction loss, 400k transitions). SAPG shows the
  *highest* reconstruction training error across widths.

---

## Evaluation

```bash
python scripts/evaluate.py --task reorientation --method sapg \
    --checkpoint runs/reorientation/sapg_seed0.pt --num-seeds 5 --paper-comparison
```

Prints a Table-1 style summary comparing the achieved metric against the paper's
reported values (mean, shaded band, delta).

---

## Tests

```bash
python -m pytest tests/ -q
```

* `test_ppo_loss.py` — Eq. 2 clipping (inside region / both clip bounds /
  negative advantage), entropy bonus Eq. 10 (leader excluded, `sigma*(j-1)`),
  bounds regularisation, module wrapper.
* `test_off_policy_loss.py` — `r` and `mu` definitions, mu-scaled clipping
  bounds, reduction to the on-policy surrogate when `i == j`, per-source
  averaging over `X = {2..M}`, `lambda` weighting, gradient flow.
* `test_importance_ratio.py` — importance-ratio contract and the `i == j`
  on-policy reduction.
* `test_critic_targets.py` — 3-step/1-step targets (gamma powers, terminal
  truncation, horizon bootstrap), value-loss scaling, `lambda' = 4.0`.

---

## Hardware / runtime

A single modern NVIDIA GPU (A100 / V100 class) with enough memory to host 24576
IsaacGym environments. Expected wall-clock per full run at the paper budget of
`2e10` transitions: 48-60 h. For development, use `--num-envs 256 --surrogate`
which reproduces the exact same code paths on CPU in seconds.

---

## Notes on paper ambiguities

* `tau = 0.95` is interpreted as the GAE lambda (the paper never defines it).
* Adam with PyTorch defaults is used as the optimizer (not stated in the paper).
* The total objective is `L_policy + lambda' * L_critic`, keeping the off-policy
  weight `lambda` (Eq. 4) separate from the critic coefficient `lambda' = 4.0`.
* The `1e-4` "bounds loss coefficient" is implemented as a quadratic penalty on
  pre-tanh outputs outside the `[-1, 1]` band (rl_games / DexPBT convention).
* `phi_j` conditioning is implemented by concatenating `phi_j` to the observation
  before the shared backbone, with `phi_j ~ N(0, I)` initialisation.
