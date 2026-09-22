# Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem

Code for reproducing **Wołczyk et al. (2024), _"Fine-tuning Reinforcement Learning Models is
Secretly a Forgetting Mitigation Problem"_** (ICML).

The paper shows that the poor *transfer* performance of RL fine-tuning is largely caused by
**forgetting of pre-trained capabilities (FPC)** — the pre-trained policy's skills on states it
used to visit collapse when the policy is fine-tuned, even though fine-tuning is supposed to only
*add* new skills. Two concrete mechanisms explain the phenomenon:

* the **state coverage gap** — the pre-trained policy rarely visits the states needed for the new
  task, so the RL objective has almost no gradient there (Appendix A.1, `two_state_mdp`);
* the **imperfect cloning gap** — fine-tuning makes the policy drift away from behaviours it was
  cloned from, and the drift is not corrected unless the pre-training states are revisited
  (Appendix A.2, `apple_retrieval`).

The proposed remedy is to treat fine-tuning as a **continual-learning** problem and to add standard
knowledge-retention techniques — **EWC**, **behavioral cloning (BC)**, **kickstarting (KS)** and
**episodic memory (EM)** — *to the actor only* (critic coefficient is always 0). This yields:

| Environment | Result |
| --- | --- |
| NetHack (Human Monk) | ~5K (π\*) → **10 588 ± 672** with KS (≈2× previous SOTA), BC ≈ 7 610, EWC ≈ 3 976, vanilla ≈ 2 664, from scratch ≈ 776 |
| Montezuma's Revenge | fine-tuning + BC > fine-tuning + EWC > from scratch > vanilla on whole-game return, with Room-7 success rate preserved |
| RoboticSequence (Meta-World) | fine-tuning + BC solves all four stages in ≈80% of runs; EM and EWC also beat vanilla; vanilla forgets the FAR stages almost entirely |

---

## 1. Repository layout

```
finetuning-rl-as-cl/
├── main.py                     # CLI dispatcher: train / evaluate / toy / analyze / test
├── train.py                    # environment-agnostic training entry point (run_env)
├── evaluate.py                 # environment-agnostic evaluation entry point
├── configs/
│   ├── nethack.yaml            # Table 1 + retention hyperparameters (EWC 2e6, BC 2.0, KS 0.5/0.99998)
│   ├── montezuma.yaml          # Table 2 + M1/M2 pipeline + Figure 13 KL sweep grid
│   ├── robotic_sequence.yaml   # SAC + Table 3 + Algorithm-1 environment
│   └── toy.yaml                # toy constants (gamma, r0, r1, eps, M, c)
├── src/
│   ├── common/                 # config, seeding, logging, checkpointing
│   ├── retention/              # ewc.py, fisher.py, behavioral_cloning.py,
│   │                           # kickstarting.py, episodic_memory.py
│   ├── nethack/                # model, encoders, dataset, env, appo_runner,
│   │                           # pretrain_baseline, per_level_eval, train_nethack
│   ├── montezuma/              # env, model, ppo_rnd, m1_train, m2_bc, train_montezuma
│   ├── robotic_sequence/       # env (Algorithm 1), model, heads, sac, train_robotic
│   ├── toy/                    # two_state_mdp.py, apple_retrieval.py
│   └── analysis/               # cka, forward_transfer, loglikelihood, pca_viz,
│                               # density_plots, return_distribution, plotting
├── scripts/
│   ├── download_nld_aa.sh      # 16 NLD-AA shards + sqlite registration (nld-aa-v0)
│   ├── download_pretrained_ckpt.sh  # NetHack 30M LSTM checkpoint (π*)
│   └── run_all.sh              # end-to-end driver (toy → robotic → montezuma → nethack → analysis)
├── tests/                      # smoke/unit tests for every component
└── requirements.txt
```

The priority ordering used during development is: retention losses → RoboticSequence → toy
examples → Montezuma → NetHack → analysis.

---

## 2. Installation

Python ≥ 3.9 (3.10 recommended).

```bash
git clone <this-repo> && cd finetuning-rl-as-cl

# Core stack (sufficient for the retention losses, toy examples and analysis)
pip install -r requirements.txt
```

Environment-specific dependencies (all optional — every trainer falls back to a deterministic
CPU **stub** when the real backend is missing):

```bash
# NetHack Learning Environment (native extension, needs a build toolchain)
pip install git+https://github.com/heiner/nle.git

# APPO reference implementation used for NetHack
git clone https://github.com/alex-petrenko/sample-factory

# AutoAscend expert (branch jt-nld) — generates the per-level saves of Section 5
git clone -b jt-nld https://github.com/cdmatters/autoascend
export AUTOASCEND_PATH=/path/to/autoascend

# Montezuma: Atari backends
pip install gymnasium ale-py          # atari-py only on Python < 3.11

# RoboticSequence: Meta-World v2
pip install metaworld
```

### Artifacts

```bash
scripts/download_nld_aa.sh --register      # 16 shards -> data/nld-aa, registered as nld-aa-v0
scripts/download_pretrained_ckpt.sh        # π* -> data/checkpoints/nethack_challenge_30M.pt
```

Both scripts are idempotent, resumable and verify their artifacts. The NLD-AA dataset supplies
~8 000 Human Monk games and is used for (a) the BC state buffer, (b) the 10 000 Fisher batches,
and (c) expert log-likelihood computations.

---

## 3. Quick start (CPU only, ~minutes)

Everything below runs without NLE / Atari / Meta-World thanks to the `--stub` fallbacks.

```bash
# 1. Unit tests for all four retention losses + Fisher estimator
python -m tests.test_retention_losses
python -m tests.test_robotic_sequence
python -m tests.test_toy_mdp
python -m tests.test_forward_transfer
python -m tests                       # runs the whole suite

# 2. Toy examples from Appendix A (no external data at all)
python -m src.toy                     # both Appendix-A sanity checks
python -m main.py toy                 # same, via the dispatcher
python -m main.py toy --sweep         # also sweep M and c (Figures 10-11)

# 3. Smoke-train RoboticSequence with each retention method (stub Meta-World)
python -m main.py train --env robotic_sequence --method none --stub --total-steps 4096
python -m main.py train --env robotic_sequence --method bc   --stub --total-steps 4096

# 4. Full smoke pipeline for every environment
scripts/run_all.sh --smoke-test
```

`scripts/run_all.sh --smoke-test` forces the stub backends, `--total-steps 4096` and a single seed;
it is the recommended way to validate an installation.

---

## 4. Retention losses (`src/retention/`)

All four techniques are applied to the **actor only**; the critic coefficient is always `0.0`.

| Method | Loss | Coefficient (NetHack) | Coefficient (Meta-World) |
| --- | --- | --- | --- |
| `ewc` | `L_aux(θ) = Σ_i F^i (θ_pre^i − θ^i)²` (Eq. 1) | `2e6` (Fisher from 10 000 NLD-AA batches) | `100` |
| `bc` | `L_BC(θ) = E_{s~B_BC}[ D_KL^s(π_θ ‖ π_*) ]` (Appendix C.2) | `2.0`, no decay, 10 000-state buffer | `1.0`, 10 000-state buffer |
| `ks` | `L_KS(θ) = E_{s~B_θ}[ D_KL^s(π_* ‖ π_θ) ]` (Appendix C.2) | `0.5` × `0.99998^t` (per train step) | `1.0` |
| `em` | *no auxiliary loss*: 10 % of the replay buffer is locked to `π_*` transitions (Appendix C.3) | — | 10 % of 100 k buffer |

Implementation notes:

* `fisher.py` supports the **expert/offline** mode (stored NLD-AA actions, NetHack) and the
  **policy/online** mode (SAC rollouts of `π_*`, following Wołczyk et al. 2021).
* `behavioral_cloning.py` exposes `CoefficientSchedule`, shared with `kickstarting.py`; BC uses
  `kind="none"` while KS uses `kind="exponential"` with `decay=0.99998`.
* `episodic_memory.py` protects the first `round(fraction · capacity)` slots and its
  `MixedBatchSampler` guarantees every batch contains prior-task transitions; `penalty()` returns a
  differentiable zero so the SAC/APPO update code is uniform across methods.
* Entropy regularisation is **disabled** whenever a retention method is active (Appendix B.1).

Quick reference:

```python
from src.retention import build_retention, retention_config

retention_config("nethack", "ks")     # {'coef': 0.5, 'decay': 0.99998, ...}
loss_fn = build_retention("ewc", actor, env_name="nethack", fisher=fisher)
total_loss = rl_loss + loss_fn.penalty_loss()
```

---

## 5. Environment-specific reproduction

### 5.1 RoboticSequence (Meta-World) — cheapest, CPU only

Implements **Algorithm 1**: four stages `hammer → push → peg-unplug-side → push-wall`, advancing on
the stage success signal, resetting the per-stage timestep counter, terminating on success or the
time limit `T = 200`, appending the normalised timestep `t/T` to the observation and granting the
augmented success reward `r'_t = β · r_t · (T − t)` with `β = 1.5`. `{hammer, push}` are CLOSE
(pre-training distribution), `{peg-unplug-side, push-wall}` are FAR.

SAC hyperparameters (Appendix B.3): 4 × 256 MLP, Leaky-ReLU, LayerNorm after the first layer, one
head per stage, automatic entropy tuning, Adam `lr = 1e-3`, batch size 128.

```bash
# π* pre-training (last two = FAR stages by default; use pretrain.scope=all_stages for the variant)
python -m src.robotic_sequence.train_robotic --config configs/robotic_sequence.yaml --pretrain-only

# Fine-tuning variants (evaluate at least 20 seeds and report 90% CIs)
for m in none ewc bc em scratch; do
  python -m src.robotic_sequence.train_robotic --method $m --seed 0 \
      --output-dir results/robotic_sequence/$m
done

# Evaluation / Table 6 forward transfer / Figure 8 log-likelihood
python -m evaluate --env robotic_sequence --method bc --checkpoint <ckpt>
python -m src.analysis.forward_transfer --results-dir results/robotic_sequence
python -m src.analysis.loglikelihood   --config configs/robotic_sequence.yaml
python -m src.analysis.cka             --reference <pistar.pt> --checkpoint <ckpt>
python -m src.analysis.pca_viz         --config configs/robotic_sequence.yaml
```

Deliverables: Figure 3c, Figure 7, Figure 8, Figure 20, Figure 22, Figures 26–27, Table 6.
Forward transfer is `(AUC − AUC^b) / (1 − AUC^b)` over prefix-task lengths `1..4`.

### 5.2 Montezuma's Revenge

* **M1** — PPO + Random Network Distillation trained from scratch until the windowed episode return
  reaches ≈ 7000 (Table 2: 128 envs, `γ = 0.999`, `γ_int = 0.99`, `λ = 0.95`, `lr = 1e-4`,
  sticky actions `p = 0.25`, max 4 500 steps/episode, RND feature dim 512, `UpdateProportion 0.25`).
* **M2 (π\*)** — behavioural cloning on **500 trajectories collected from Room 7 onward**
  (room completion = earning a coin, acquiring an item, or exiting through a different passage).
* **Fine-tuning** — whole-game PPO + RND with `method ∈ {none, bc, ewc}`; the from-scratch baseline
  never sees BC data. The BC KL weight is not stated in the paper, so it is selected via the
  Figure-13 sweep over `{0.1, 0.5, 1.0, 2.0, 5.0}`.

```bash
python -m src.montezuma.m1_train          --config configs/montezuma.yaml
python -m src.montezuma.m2_bc             --config configs/montezuma.yaml --mode collect
python -m src.montezuma.m2_bc             --config configs/montezuma.yaml --mode pretrain
for m in none bc ewc scratch; do
  python -m src.montezuma.train_montezuma --method $m --seed 0
done
python -m src.montezuma.m2_bc --mode sweep-kl        # Figure 13
python -m evaluate --env montezuma --method bc --checkpoint <m2.pt>
```

Room-7 success rate and room visitation are logged every 5 M steps.
Deliverables: Figure 3b, Figure 6, Figure 13, Figures 17–19.

### 5.3 NetHack (Human Monk)

Model (Appendix B.1): main-screen character + colour embedding lookup → ResNet, 2-layer MLP for
`blstats`, 2-layer MLP for the message buffer, merged features into an LSTM of width **1738**, then
policy (120 actions) and baseline heads. The released 30 M LSTM checkpoint is the initialisation;
**encoders are frozen** during fine-tuning, and the baseline head is pre-trained for 500 M steps
with everything else frozen.

APPO fine-tuning (Table 1): Adam `lr = 1e-4` (`β = (0.9, 0.999)`, `eps = 1e-7`, weight decay `1e-4`),
unroll 32, batch 128, `γ = 0.999999`, entropy cost `0.001`, gradient-norm clip 4, reward clip 10,
APPO clip 0.1 (policy) / 1.0 (baseline), `λ = 0.95`. Run for ≥ 500 M environment steps,
checkpointing every 25 M.

```bash
# 0. verify π* (~5K Human Monk)
python -m src.nethack.train_nethack --verify-pi-star --config configs/nethack.yaml

# 1. baseline-head pre-training (500M steps, encoders + policy frozen)
python -m src.nethack.pretrain_baseline --config configs/nethack.yaml

# 2. diagonal Fisher from 10000 NLD-AA batches + BC state buffer (10000 samples)
python -m src.nethack.train_nethack --compute-fisher --build-bc

# 3. the five variants
for m in scratch none ewc bc ks; do
  python -m src.nethack.train_nethack --method $m --seed 0 --total-steps 500000000
done

# 4. per-level (level 4 / Sokoban) evaluation, 200 AutoAscend saves per level, every 25M steps
python -m src.nethack.per_level_eval --generate-saves --level level_4 --num-saves 200
python -m src.nethack.per_level_eval --generate-saves --level sokoban --num-saves 200
python -m evaluate --env nethack --checkpoint <ckpt> --per-level --levels level_4,sokoban
```

Evaluation rollouts stop at death, after **150 steps without progress** (progress = score, or
dungeon depth), or at 100 000 steps. Full evaluation is 1 000 episodes every 25 M steps.
Deliverables: Figure 3a, Figure 5, Table 4, Table 5.

### 5.4 Toy examples (Appendix A)

```bash
python -m src.toy.two_state_mdp      # closed-form v0(theta)
python -m src.toy.apple_retrieval    # REINFORCE on the 1-D gridworld
```

* Two-state MDP: gradient-ascent fine-tuning converges to `θ = 0.11 → v0 = 2.22`
  (state coverage gap) and `θ = 0.08 → v0 = 9.93` (imperfect cloning gap), both below the global
  optimum `v0(1) = 10`.
* `AppleRetrieval` uses the two-parameter policy `π_{w,b}(o) = σ(w·o + b)`, Phase-2 pre-training and
  full-task fine-tuning; forbidding the updating of `b` (the *state coverage gap* remedy) keeps the
  Phase-2 behaviour. Sweeps over `M` and `c` reproduce Figures 10–11.

---

## 6. Analysis modules (`src/analysis/`)

| Module | Reproduces |
| --- | --- |
| `cka.py` | layer-wise CKA drift (later policy layers change more, partial recovery as FAR tasks return) — Figures 26–27 |
| `forward_transfer.py` | `FT = (AUC − AUC^b)/(1 − AUC^b)` over prefix-task lengths — Table 6 |
| `loglikelihood.py` | expert-action log-likelihood every 50 k steps (collapse ≈ 100 k, no full recovery) — Figure 8 |
| `pca_viz.py` | PCA of the state space coloured by log-likelihood — Figure 8 |
| `density_plots.py` | NetHack dungeon-level visitation density — Figure 5 |
| `return_distribution.py` | per-method return distributions + paper-ordering checks — Figure 3 |
| `plotting.py` | shared curve/CI/heatmap utilities (90 % CIs) |

```bash
python -m main.py analyze --what all --results-dir results
python -m main.py analyze --what forward_transfer
```

---

## 7. Expected results (reproduction checklist)

| Quantity | Paper value | Where |
| --- | --- | --- |
| π\* Human Monk score | ≈ 5 000 | `--verify-pi-star` |
| from-scratch APPO | ≈ 776 | `none`/`scratch` variants |
| vanilla fine-tuning | collapses on FAR levels | Figure 3a, Table 5 |
| + EWC (2e6) | ≈ 3 976 | Figure 3a |
| + BC (2.0) | ≈ 7 610 | Figure 3a |
| + KS (0.5, 0.99998) | **10 588 ± 672** | Figure 3a |
| Montezuma M1 return | ≈ 7 000 | `m1_summary.json` |
| RoboticSequence + BC | all four stages ≈ 80 % | Figure 7 |
| Table 6 forward transfer | fine-tuning 0.30→0.00; EWC 0.85→0.75; BC ≈ 0.95 | `compute_table` |

Every retention method must reduce FAR-state degradation relative to vanilla, with the ordering
`KS > BC > EWC > vanilla` (NetHack), `BC > EWC > none > scratch` (Montezuma) and
`BC > EM > EWC > vanilla` (RoboticSequence). `evaluate.py` and `train.py` embed these expectations in
`PAPER_REFERENCE` and report an automatic ordering check.

Outputs are written as JSON (`summary.json`, `nethack_summary.json`, `montezuma_summary.json`,
`m1_summary.json`, `kl_weight_sweep.json`, `density.json`, `per_level_eval.json`, …) under
`results/<env>/<method>/seed_<k>/`, which all analysis modules discover automatically.

---

## 8. Hardware

| Environment | Requirement |
| --- | --- |
| Retention losses, toy, analysis | CPU only (minutes) |
| RoboticSequence | CPU, 8 cores / 30 GB RAM per experiment, ~48 h per run (≥ 20 seeds) |
| Montezuma | GPU with ≥ 16 GB VRAM (128 parallel envs) |
| NetHack | NVIDIA A100-class GPU; ≥ 500 M env steps per variant (< 24 h) |

---

## 9. Notes, ambiguities and reproduction gaps

* **Actor-only retention.** Every auxiliary term is added to the actor loss only; the critic
  coefficient is fixed at `0.0` in all configs.
* **RoboticSequence pre-training scope.** §3/B.3 say π\* is pre-trained on the **last two (FAR)**
  stages, while the addendum also mentions all stages. Both are implemented
  (`pretrain.scope: last_two|all_stages`); the last-two model is π\* for the main results.
* **Montezuma BC KL weight** is not stated in the paper — the Figure-13 sweep provides it.
* **Encoder widths** are not published; the released Tuyls et al. (2023) checkpoint is reused and
  the LSTM width follows Table 1 (`hidden_dim = 1738`).
* **NetHack training length** is not fixed by the paper; ≥ 500 M steps with 25 M-step checkpoints.
* **RoboticSequence alternative sequences** follow the Continual World (Wołczyk et al. 2021) naming.
* Runs that could not complete within the available compute budget are labelled
  `"reproduction-gap"` in their summaries rather than blocking the pipeline; `scripts/run_all.sh`
  records per-stage failures and continues.
* All heavy dependencies (`torch`, `nle`, `gymnasium`, `metaworld`, `matplotlib`) are imported
  lazily, and every trainer supports `--stub`, so the repository is importable and testable on a
  bare CPU machine.
