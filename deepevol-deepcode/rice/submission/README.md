# RICE: A Refining Scheme for Reinforcement Learning with Explanation

Reference implementation of **RICE** (ICML 2024, PMLR 235), a two-stage scheme that
*refines* a pre-trained, locally-optimal (bottlenecked) deep RL policy:

1. **Stage 1 — Explanation.** A **mask network** `~π_θ(a_t^m | s_t)` (a simplified
   StateMask) is trained with **vanilla PPO plus a blinding bonus** `α·a_t^m` to assign a
   *step-level importance score* to every visited state. The importance of a state is

   ```
   I(s_t) = P(a_t^m = 0 | s_t)      # probability the mask says "keep"
   ```

   The executed action follows the masked-action operator from the paper:

   ```
   a = a_t   if a_t^m = 0   (keep the target policy's action)
   a = random if a_t^m = 1   (blind: take a uniformly random action)
   ```

2. **Stage 2 — Refinement.** The frozen pre-trained policy `π` is refined with PPO on a
   **mixed initial state distribution**

   ```
   μ(s) = β · d_ρ^{π̂}(s) + (1 − β) · ρ(s)
   ```

   realised via Algorithm 2's reset rule (reset to the mask-identified critical state with
   probability `p ≡ β`, otherwise sample `s_0 ~ ρ`), plus a **Random Network Distillation**
   intrinsic reward

   ```
   R_t^RND = || f(s_{t+1}) − f̂(s_{t+1}) ||² ,     R'_t = R_t + λ · R_t^RND
   ```

Paper-specific hyperparameters and the constants used when the paper is silent are listed
in [Hyperparameters](#hyperparameters) and in `configs/*.yaml`.

> **Reproduction scope.** Validation targets **trends / takeaways** (qualitative ordering and
> effect directions), not exact numbers. Out of scope: §3.4 theory (Appendix B),
> SparseWalker2d refining, sparse-MuJoCo sensitivity, autonomous-driving qualitative
> analysis, and all Malware Mutation experiments. The Table-1 claim that "Ours explanation
> strictly beats StateMask" is intentionally **not** enforced (they are comparable).

---

## Repository layout

```
rice/
  envs/            environment factory + sparse reward + reset (Go-Explore style) + obs normalizer
    make_env.py       Hopper-v3, Walker2d-v3, Reacher-v2, HalfCheetah-v3, SparseHopper,
                      SparseHalfCheetah, SelfishMining-v0, Cage2-v0, MetaDrive-Macro-v1
    sparse_reward.py  sparse reward (forward-x threshold) wrappers
    reset_wrapper.py  state snapshot / restore + deterministic action replay
    normalizer.py     observation normalization (Walker2d, HalfCheetah)
  models/
    policies.py       per-app target policy π and mask-net architectures
  explanation/     Stage 1 (Algorithm 1)
    mask_network.py   mask net module + masked-action operator
    mask_trainer.py   Algorithm 1: vanilla PPO + blinding bonus α·a_t^m
    importance.py     state importance I(s) = P(keep | s)
    critical_state.py critical-state selection (roll π for K, take argmax importance)
    fidelity.py       sliding-window fidelity score (Experiment I)
  refining/        Stage 2 (Algorithm 2)
    rnd.py            RND target f, predictor f̂, normalised intrinsic bonus
    mixed_init.py     mixed initial state distribution sampler μ(s)
    ppo_refine.py     PPO refinement loop with mixed init + RND
  baselines/       comparison methods
    random_explanation.py   Random explanation
    statemask_r.py          StateMask (primal-dual) + StateMask-R refining baseline
    integrated_gradients.py Integrated Gradients explanation
    airs.py                 AIRS explanation
    ppo_finetune.py         PPO fine-tuning refining baseline
    jsrl.py                 JSRL refining baseline
    sac_finetune.py         SAC fine-tuning (Experiment IV)
    gail.py                 GAIL approximation of a SAC expert (Experiment IV)
    sil.py                  Self-Imitation Learning refining baseline (Table 5)
  utils/           seeding, io, logging, metrics (fidelity score, running mean/std)
configs/           default.yaml + per-application overrides
scripts/           train_target.py, train_mask.py, run_fidelity.py, run_refine.py,
                   run_ablation.py, plot_results.py
experiments/       exp1_fidelity_efficiency.py ... exp5_hyperparams.py
tools/             checkpoint_manager.py
main.py            top-level CLI dispatcher
```

Architectures (per addendum "Architectures"): Stable-Baselines3 default `MlpPolicy`
(2×64 tanh) for all dense/sparse MuJoCo environments; Selfish Mining
`[128,128,128,128]`; CAGE Challenge 2 `[64,64,64]`; MetaDrive DI-engine default VAC.

---

## Installation

Python 3.8–3.10.

```bash
pip install -r requirements.txt        # full stack: torch, SB3, gym, mujoco, pandas, matplotlib
# or
pip install -e .                       # core package (numpy + PyYAML only)
pip install -e ".[rl,mujoco,plot]"     # add RL + MuJoCo + plotting extras
```

Heavy/optional stacks (present but commented in `requirements.txt`):
`di-drive` + `metadrive-simulator` (autonomous driving), `tianshou` (Malware Mutation —
**out of scope**), `captum` (Integrated Gradients, optional).

The package degrades gracefully: every `rice.*` sub-module guards its imports, so `main.py
--help`, config loading, and the result/plotting tooling work even without torch / SB3 /
MuJoCo installed.

---

## Quick start

```bash
# 1. Inspect the setup (no training)
python main.py check
python main.py list-envs
python main.py show-config --config hopper
python main.py show-table3

# 2. Pre-train the frozen target policy π  (Stage 0)
python main.py train-target --env hopper

# 3. Train the Stage-1 mask explanation (Algorithm 1) + StateMask timing baseline
python main.py train-mask --env hopper --methods ours,statemask

# 4. Experiment I: fidelity across K ∈ {10,20,30,40}% and mask-training efficiency
python main.py fidelity --env hopper

# 5. Experiment II: refine with Ours vs PPO fine-tune / StateMask-R / JSRL
python main.py refine --env hopper --methods ours,ppo_finetune,statemask_r,jsrl

# 6. Run whole experiments (I–V) and ablations
python main.py experiment --name exp1 --envs hopper,walker2d
python main.py ablation --ablation all --envs hopper
python main.py run-all --envs hopper

# 7. Build tables/figures from the JSON artifacts in results/
python main.py plot
```

Equivalent entry points are exposed as console scripts after `pip install -e .`:
`rice`, `rice-train-target`, `rice-train-mask`, `rice-fidelity`, `rice-refine`,
`rice-ablation`, `rice-plot`.

---

## Pipeline details

### Stage 0 — Pre-trained target policy π (`scripts/train_target.py`)

Trains (or loads) the sub-optimal/bottlenecked policy π that Stage 1 explains and Stage 2
refines. Backends: SB3 `PPO` (preferred) → native PyTorch PPO from `rice/refining/ppo_refine.py`
with `use_mixed_init=False, use_rnd=False, p=0, λ=0` → untrained fallback. Checkpoints are
written to `policies/<env>_ppo[_seedN].zip`; π is **frozen** afterwards. Table-1 "No Refine"
reference returns (Hopper 3559.44, Walker2d 3339.68, Reacher −5.51, HalfCheetah 4540.50,
CAGE-2 −23.64, Auto Driving 10.30) are stored for trend checks only.

### Stage 1 — Explanation (`rice/explanation/`, Algorithm 1)

`MaskTrainer` (`mask_trainer.py`) implements Algorithm 1:

```
for iteration:
    s_0 ~ ρ ; θ_old ← θ
    for t = 0..T:
        a_t ~ π(·|s_t)                # frozen target policy
        a_t^m ~ ~π_{θ_old}(·|s_t)     # mask action (Bernoulli / 2 logits)
        a = a_t ⊙ a_t^m               # masked-action operator
        (s_{t+1}, R_t) = env.step(a)
        store (s_t, s_{t+1}, a_t^m, R'_t) with R'_t = R_t + α·a_t^m
    update θ with VANILLA PPO on D   # max η(π̄), no primal-dual
```

Because the objective is `max η(π̄)` (Theorem 3.3) instead of StateMask's primal-dual
`min |η(π) − η(π̄)|`, masking trains faster — the **~16.8 %** wall-clock reduction reported
in Table 4 (e.g. Hopper 12426 s vs 15393 s; HalfCheetah 1317 s vs 1579 s; CAGE-2 65400 s vs
79382 s). The StateMask timing baseline lives in `rice/baselines/statemask_r.py`
(`StateMaskTrainer`, projected dual ascent on α).

State importance is `I(s) = P(a_t^m = 0 | s)` (`importance.py`); critical states are the
argmax-importance states of a length-`K` roll of the frozen π (`critical_state.py`).

### Fidelity score (Experiment I, `rice/explanation/fidelity.py`)

```
l = L × K                           # sliding-window width, K ∈ {10,20,30,40}%
window = argmax_w mean(I over w)    # highest average importance window
fast-forward to window start → random actions for l steps → resume π to episode end
d = |R' − R| ;  d_max = env max single-episode reward
fidelity = log(d / d_max) − log(l / L)          # higher is better
```

Averaged over **500 trajectories × 3 seeds**, per environment, reporting mean ± std.

### Stage 2 — Refinement (`rice/refining/`, Algorithm 2)

`PPORefiner` (`ppo_refine.py`) implements Algorithm 2:

```
for iteration:
    RAND_NUM ~ U(0,1)
    if RAND_NUM < p:                        # s_0 ~ d_ρ^{π̂}  (critical state)
        roll frozen π for K steps → mask argmax → reset env to that state
    else:                                   # s_0 ~ ρ
        env.reset()
    roll trainable π_θ for T steps, sampling a_t ~ π_θ,
      stepping env with R_t + λ·||f(s_{t+1}) − f̂(s_{t+1})||²
    collect D = {(s_t, s_{t+1}, a_t, R_t + λ·R_t^RND)}
    optimize π_θ with the standard PPO clipped loss on D
    update f̂ with MSE (Adam)
π' ← π_θ
```

* `rnd.py`: frozen random target `f` + trainable predictor `f̂` (small `[64,64]` MLP),
  intrinsic reward `||f − f̂||²` normalised by the **running std** of the prediction error
  (paper-unspecified default); predictor updated with MSE / Adam.
* `mixed_init.py`: `p` plays the role of the mixture weight β. `p=0` ⇒ pure `ρ`,
  `p=1` ⇒ always reset to the critical state.
* PPO hyperparameters where the paper is silent: SB3 defaults
  (γ = 0.99, GAE λ = 0.95, clip = 0.2, lr = 3e-4, n_epochs = 10, batch = 64).

---

## Baselines

**Explanations** (fix refine = Ours, vary explanation — Experiment III):

| Method | Module |
| --- | --- |
| Random | `rice/baselines/random_explanation.py` |
| StateMask | `rice/baselines/statemask_r.py` (`StateMaskExplainer`, primal-dual) |
| Integrated Gradients | `rice/baselines/integrated_gradients.py` |
| AIRS | `rice/baselines/airs.py` |
| **Ours (RICE mask net)** | `rice/explanation/` |

**Refining methods** (fix explanation = Ours, vary refiner — Experiment II):

| Method | Module |
| --- | --- |
| PPO fine-tuning (lower LR, continue PPO) | `rice/baselines/ppo_finetune.py` |
| StateMask-R (always reset to critical state + fine-tune) | `rice/baselines/statemask_r.py` |
| JSRL (guided/exploration policy curriculum) | `rice/baselines/jsrl.py` |
| SAC fine-tuning (Experiment IV) | `rice/baselines/sac_finetune.py` |
| GAIL (learn an approximate policy from a SAC expert) | `rice/baselines/gail.py` |
| SIL (Self-Imitation Learning, Table 5) | `rice/baselines/sil.py` |
| **Ours (mixed init + RND)** | `rice/refining/ppo_refine.py` |

All refining baselines use the **same explanation** as RICE where one is needed, to keep the
comparison fair.

---

## Environments

| Config | Gym id | Notes |
| --- | --- | --- |
| `hopper.yaml` | `Hopper-v3` | dense MuJoCo |
| `walker2d.yaml` | `Walker2d-v3` | dense, observation normalisation |
| `reacher.yaml` | `Reacher-v2` | dense, short horizon (50 steps) |
| `halfcheetah.yaml` | `HalfCheetah-v3` | dense, observation normalisation |
| `sparse_hopper.yaml` | `SparseHopper-v0` | sparse reward: forward-x only if `x > 0.6` |
| `sparse_halfcheetah.yaml` | `SparseHalfCheetah-v0` | sparse reward: forward-x only if `x > 5.0` |
| `selfish_mining.yaml` | `SelfishMining-v0` | 3 discrete actions (Adopt / Reveal / Mine) |
| `cage2.yaml` | `Cage2-v0` | blue-agent action set, Restore penalty −1, trails {30,50,100}, final reward = sum of the three average rewards |
| `autodriving.yaml` | `MetaDrive-Macro-v1` | 2-D continuous action → steering / acceleration / brake |

Malware Mutation is explicitly **excluded** (`main.py list-envs` marks it out of scope).

---

## Experiments

| Driver | Paper experiment | What it does |
| --- | --- | --- |
| `experiments/exp1_fidelity_efficiency.py` | Experiment I (Table 4) | Fidelity for `K ∈ {10,20,30,40}%` over 500 trajectories × 3 seeds for Ours / StateMask / Random, plus mask-training wall-clock (expected ~16.8 % reduction). |
| `experiments/exp2_refine_effectiveness.py` | Experiment II (Table 1 left, Fig. 2) | Refine with Ours vs PPO fine-tune / StateMask-R / JSRL; dense applications and sparse Hopper / Sparse HalfCheetah. |
| `experiments/exp3_explanation_quality.py` | Experiment III (Table 1 right, Table 6) | Fix refine = Ours, vary explanation ∈ {Random, StateMask, Ours} (+ IG / AIRS). |
| `experiments/exp4_sac_agent.py` | Experiment IV (Fig. 3) | Pre-train SAC on Hopper, GAIL-approximate, refine with RICE vs PPO fine-tune / StateMask-R / JSRL / SAC fine-tuning. |
| `experiments/exp5_hyperparams.py` | Experiment V (Figs. 7–9) | Sensitivity sweeps over `p ∈ {0,0.25,0.5,0.75,1}`, `λ ∈ {0,0.1,0.01,0.001}`, `α ∈ {0.01,0.001,0.0001}`. |

Run them directly or through `main.py`:

```bash
python experiments/exp1_fidelity_efficiency.py --envs hopper,walker2d
python experiments/exp2_refine_effectiveness.py --envs hopper --methods ours,ppo_finetune,statemask_r,jsrl
python experiments/exp3_explanation_quality.py --envs hopper --explanations random,statemask,ours
python experiments/exp4_sac_agent.py --env hopper
python experiments/exp5_hyperparams.py --envs hopper --sweeps p,lambda,alpha
```

Artifacts are written to `results/<experiment>/<experiment>_<env>.json` (+ `.txt`
summaries). `scripts/plot_results.py` scans those JSON files and renders Table 1 / 4 / 5 / 6
and Figures 2, 3, 7, 8, 9 into CSV / Markdown / PDF under `results/plots`.

### Expected trends (success criteria)

* **Experiment I.** Ours fidelity ≈ StateMask, both clearly **> Random**; mask-training time
  for a fixed sample budget is **~16.8 % lower** than StateMask.
* **Experiment II.** Ours yields the best final reward across Hopper / Walker2d / Reacher /
  HalfCheetah / Selfish Mining / CAGE-2 / Auto Driving and the highest refining efficiency on
  Sparse Hopper / Sparse HalfCheetah; PPO fine-tuning is only marginally better than no
  refinement; StateMask-R does not always help (over-sensitive to the initialization
  distribution). Reference trends e.g. Hopper 3559.44 → 3663.91, CAGE-2 −23.64 → −20.02,
  Auto Driving 10.30 → 17.03.
* **Experiment III.** Ours and StateMask comparable, both **> Random**. `Ours > StateMask` is
  **not** required.
* **Experiment IV.** RICE beats PPO fine-tuning / StateMask-R / JSRL and SAC fine-tuning after
  switching a pre-trained SAC agent to PPO; the PPO switch breaks the bottleneck.
* **Experiment V.** `p = 0` (all `ρ`) and `p = 1` (all critical) are worse than the mixed
  distribution; `p = 0.25 / 0.5` are best. `λ > 0` improves over `λ = 0`; performance is
  largely insensitive to `λ`, with `λ = 0.01` generally good. Fidelity is insensitive to `α`.

Trend checks are **advisory** — `scripts/run_ablation.py` and the experiment drivers always
exit 0, and `scripts/plot_results.py` reports a `trend_report` dict rather than failing.

---

## Hyperparameters

Per-application defaults (Table 3) are consolidated in `main.py` (`TABLE3`) and
`configs/*.yaml`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `p` (= β) | 0.5 | probability of resetting to a critical state in μ(s) |
| `λ` | 0.01 | RND intrinsic-reward coefficient |
| `α` | 1e-4 | blinding-bonus coefficient (Table 3). §4.3 / C.3 prose says 0.01; `α` is treated as **insensitive** and exposed for the sweep. |

Values chosen where the paper is silent (documented in the configs):

* PPO: γ = 0.99, GAE λ = 0.95, clip = 0.2, lr = 3e-4, n_epochs = 10, batch = 64.
* Algorithm-2 trajectory length `K` and PPO `n_steps` default to the episode horizon `T`.
* RND: normalisation = divide by running std of the prediction error; net sizes = small
  `[64,64]` MLP; predictor update = Adam + MSE.
* `d_max` = per-environment maximum single-episode reward (fidelity normaliser).
* Observation normalization only for Walker2d and HalfCheetah (Appendix C.2).
* Mask-training sample budgets: 300 000 for MuJoCo, 1.5 M Selfish Mining, 10 M CAGE-2,
  2 443 260 Auto Driving (Table 4).

---

## Reproducing the paper

```bash
# full pipeline across the dense applications
python main.py run-all --envs hopper,walker2d,reacher,halfcheetah --seeds 0,1,2

# experiments individually
for e in exp1 exp2 exp3 exp4 exp5; do python main.py experiment --name $e; done

# ablations (explanation / refining / hyperparams / SIL)
python main.py ablation --ablation all --envs hopper,walker2d

# tables and figures
python main.py plot --formats csv,md,pdf
```

Outputs:

* `policies/` — checkpoints (`<env>_ppo.zip`, `<env>_mask.pt`, `<env>_refined.pt`) plus
  `registry.json`.
* `results/target`, `results/mask`, `results/fidelity`, `results/refine`, `results/exp1..5`,
  `results/ablation` — JSON reports and text summaries.
* `results/plots/` — CSV / Markdown tables and PDF figures.
* `logs/` — per-run `log.txt`, `config.json`, `progress.json`.

Checkpoint handling is centralised in `tools/checkpoint_manager.py`
(`python tools/checkpoint_manager.py list|index|summary|resolve|verify|prune`), so Stage 1 can
reuse the Stage-0 policy and Stage 2 can reuse the Stage-1 mask without re-training.

---

## Notes on faithful reproduction

* Algorithm 1 is deliberately **vanilla PPO** with an augmented reward `R'_t = R_t + α·a_t^m`
  (no primal-dual), matching Theorem 3.3's reformulation — this is the source of the
  training-time gain over StateMask.
* `p` and `β` are the same knob; `p=0` reproduces plain PPO on `ρ`, `p=1` reproduces
  StateMask-R's "always critical" behaviour.
* Random-explanation runs intentionally carry `mask_net=None`, which the whole pipeline
  interprets as uninformative/random scoring.
* The Docker/CPU path is fully supported for the MuJoCo experiments; the authors used 8×A100
  but only trends are required here.

## Citation

```bibtex
@inproceedings{cheng2024rice,
  title     = {RICE: A Refining Scheme for Reinforcement Learning with Explanation},
  author    = {Cheng, Zelei and others},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  series    = {PMLR},
  volume    = {235},
  year      = {2024}
}
```

## License

Reference implementation for academic reproduction. See the original paper and the
official repository <https://github.com/chengzelei/RICE> for authoritative details.
