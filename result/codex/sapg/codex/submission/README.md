# SAPG: Split and Aggregate Policy Gradients - reproduction

Clean-room re-implementation of

> **SAPG: Split and Aggregate Policy Gradients** - Jayesh Singla\*, Ananye Agarwal\*, Deepak Pathak.
> Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235.

The paper argues that on-policy RL saturates when the number of parallel
environments grows, because i.i.d. Gaussian sampling makes a large fraction of
the "parallel" data redundant, and proposes to *split* the environments between
`M` policies that are diversified (latent conditioning + entropy
regularisation) and to *aggregate* their data into a single leader policy with
an importance-sampled (off-policy) PPO update.

Everything in this repository was written from `paper/paper.md` plus the
clarifications in `paper/addendum.md`; the paper's own code base was not used.

---

## 1. What is implemented

| Paper item | Where |
| --- | --- |
| **SAPG algorithm** (Algorithm 1, Sec. 4.6): split environments into `M` blocks, per-policy rollouts `D_1..D_M`, leader updated with its own data + sub-sampled off-policy data from the followers | `sapg/algorithms/sapg.py` |
| **Off-policy importance-sampled PPO loss** (Eq. 3, Sec. 4.1), `r = pi_i/pi_j`, `mu = pi_{i,old}/pi_j`, clip range `mu(1-eps)..mu(1+eps)` | `sapg/algorithms/losses.py` |
| **On-policy PPO loss** (Eq. 2) with 3-step critic targets (Eq. 5) and GAE(`tau=0.95`) advantages | `sapg/algorithms/losses.py`, `sapg/algorithms/storage.py` |
| **Off-policy 1-step value targets** (Eq. 6) for follower data | `sapg/algorithms/sapg.py` (`_build_offpolicy_dataset`) |
| **Shared backbone conditioned on per-policy `phi_j`** (Sec. 4.4 + addendum): actor `B_theta`, critic `C_psi`, `phi_j in R^32` (AllegroKuka) / `R^16` (hands) | `sapg/algorithms/actor_critic.py` |
| **Leader / follower aggregation** (Sec. 4.3) and **symmetric aggregation** (Sec. 4.2) | `sapg/algorithms/sapg.py` (`aggregation: leader_follower / symmetric`) |
| **Entropy regularisation of the followers** (Sec. 4.5): follower `j` gets coefficient `sigma*j`, the leader none | `sapg/algorithms/sapg.py` (`entropy_coef_for`) |
| **Baselines**: PPO, DexPBT (population-based PPO), PQL (parallel DDPG with mixed exploration) | `sapg/algorithms/ppo.py`, `pbt.py`, `pql.py` |
| **The 5 benchmark tasks** (Sec. 5.1 / App. A): AllegroKuka Regrasping, Throw, Reorientation; ShadowHand and AllegroHand in-hand reorientation | `sapg/envs/isaacgym/` |
| **Reward terms and success-tolerance curriculum** of the hard tasks (App. A) | `sapg/envs/isaacgym/rewards.py` |
| **Ablations** (Sec. 6.3): no off-policy, symmetric, high off-policy ratio, entropy coefficients `{0, 0.003, 0.005}` | `sapg/configs/ablations/` (AllegroKuka) and `sapg/configs/ablations_inhand/` (hands), `sapg/algorithms/sapg.py` |
| **Figure 5 / Table 1**: performance curves and final metrics with standard error over seeds | `scripts/plot_results.py`, `sapg/utils/plotting.py` |
| **Figure 2**: PPO performance vs batch size with the SAPG reference line | `sapg/analysis/batch_size_study.py` |
| **Figures 7 & 8** (Sec. 6.4): PCA and MLP state-reconstruction diversity metrics | `sapg/analysis/pca_diversity.py`, `sapg/analysis/mlp_diversity.py` |
| **CPU toy suite** used to exercise every algorithm and analysis without a GPU | `sapg/envs/toy/` |

## 2. Repository layout

```
sapg/
  algorithms/   actor_critic.py storage.py losses.py runner.py
                sapg.py ppo.py pbt.py pql.py base.py
  envs/         base.py | isaacgym/{allegro_kuka,in_hand,rewards,assets}.py
                toy/multimodal_collect.py
  analysis/     pca_diversity.py mlp_diversity.py batch_size_study.py
  utils/        config.py logger.py plotting.py results.py running_stat.py
                state_dataset.py seeding.py
  configs/      base.yaml allegrokuka.yaml inhand.yaml sapg_*.yaml ppo_*.yaml
                pbt_*.yaml pql_*.yaml ablations/*.yaml toy_*.yaml toy_ablations/*.yaml
scripts/        train.py eval.py plot_results.py run_experiments.py
                run_diversity_analysis.py run_batch_size_study.py smoke_toy.sh
tests/          unit + smoke tests (importance sampling, returns, architecture,
                trainers, toy environment)
```

## 3. The SAPG update, in code

One iteration (`SAPGTrainer.update`, mirroring Algorithm 1):

1. **Split** - `MPolicyRunner.collect` steps the `N` environments; block `j`
   (of size `N/M`) is controlled by policy `j`, so every policy fills its own
   buffer `D_j` with behaviour log-probs and values.
2. **Aggregate** - for every policy `i` with non-empty source set `X_i`
   (`X_1 = {2..M}` for the leader, `X_i = {}` for the followers, or
   `X_i = {1..M}\{i}` for the symmetric variant) `_build_offpolicy_dataset`
   samples `|D_i|` transitions (whole sequences) uniformly from the union of the
   source buffers, evaluates `pi_{i,old}` and `V_{i,old}` on them *before* any
   gradient step and forms

   ```
   A_off        = r + gamma * V_{i,old}(s') * (1 - done) - V_{i,old}(s)
   V_off_target = r + gamma * V_{i,old}(s') * (1 - done)          (Eq. 6)
   mu           = pi_{i,old}(s, a) / pi_j(s, a)
   ```

3. **Update** - a single backward pass over

   ```
   L = L_on(pi_1) + lambda * L_off(pi_1; X_1)
       + sum_{j=2..M} [ L_on(pi_j) + sigma * j * H(pi_j) ]
   ```

   updates `theta` and `psi` with the gradients of *all* policies, while `phi_j`
   receives gradients only from policy `j`'s own terms (it appears nowhere
   else). `lambda = 1` (Sec. 4.3).

The Sec. 6.3 ablations are configuration switches rather than code forks:

```yaml
algo:
  aggregation: leader_follower   # leader_follower | symmetric | none (= w/o off-policy)
  use_offpolicy: true            # false -> "Ours (w/o off-policy)"
  offpolicy_subsample: true      # false -> "Ours (high off policy ratio)"
  entropy_coef: 0.0              # 0.003 / 0.005 -> entropy-regularised followers
```

## 4. Hyperparameters

Every number the paper specifies is used verbatim and lives in `sapg/configs/`:

* `M = 6` policies, `N = 24576` environments, 16 steps of experience per update,
  5 seeds with mean +/- standard error (Sec. 5.2).
* AllegroKuka (Table 2): LSTM mean network with 1 layer of 768 units behind an
  MLP `768x512x256` (ELU), input-independent learnable sigma, `gamma = 0.99`,
  `tau = 0.95`, `lr = 1e-4`, KL threshold `0.016`, grad-norm `1.0`, entropy
  coefficient `0`, clip `0.1`, mini-batch `num_envs * 4`, critic coefficient
  `4.0`, horizon `16`, LSTM sequence length `16`, bounds-loss coefficient
  `1e-4`, 2 mini-epochs, `n = 3`-step critic returns.
* ShadowHand (Table 3) / AllegroHand (Table 4): MLP `512x512x256x128` and
  `512x256x128` (ELU), `lr = 5e-4`, horizon `8`, 5 mini-epochs, clip `0.1`/`0.2`.
* `sigma` is tuned per environment over `{0, 0.003, 0.005}`; `sigma = 0` works
  best for ShadowHand, AllegroHand, Regrasping and Throw, `sigma = 0.005` for
  Reorientation (Sec. 5.2).

## 5. Running the code

### 5.1 Full-scale reproduction (GPU + IsaacGym)

The five benchmark tasks need IsaacGym and a GPU; each run collects ~2e10
transitions (48-60 h per run on a single GPU according to the paper). Point
`env.asset_root` (or the `SAPG_ASSET_ROOT` environment variable) at the
Allegro-Kuka / ShadowHand / AllegroHand asset directory (URDF + meshes, as
released with the environments the baselines use; see
`sapg/envs/isaacgym/assets.py`).

```bash
python scripts/train.py --config sapg/configs/sapg_allegrokuka_reorientation.yaml \
    --logdir runs/sapg_reorientation --seed 0

# the full matrix of the paper (5 tasks x 4 methods x 5 seeds) + ablations
python scripts/run_experiments.py --dry-run --with-ablations

# figures 5/6 and table 1 once the runs are finished
python scripts/plot_results.py --root runs --out figures --results results

# figure 2: PPO vs batch size, with the SAPG reference line
python scripts/run_batch_size_study.py --env regrasping \
    --env-counts 128 512 2048 8192 24576 --seeds 0 1 2

# figures 7 and 8: state-diversity metrics
python scripts/run_diversity_analysis.py \
    --run sapg=runs/sapg_reorientation_seed0 --run ppo=runs/ppo_reorientation_seed0 \
    --random-config sapg/configs/sapg_allegrokuka_reorientation.yaml
```

### 5.2 CPU smoke run

`sapg/envs/toy/multimodal_collect.py` is a small vectorised control task with
several **disjoint reward modes** (K landmarks that must be captured in
sequence) - exactly the structural property SAPG targets: a unimodal Gaussian
policy keeps re-visiting the mode it already found, so adding parallel
environments does not help, while `M` diversified policies plus
importance-sampled aggregation cover more of the reward landscape. It runs on
CPU in minutes, which makes the algorithm, the baselines, the ablations and both
diversity metrics testable end-to-end:

```bash
bash scripts/smoke_toy.sh 150 768       # SAPG, PPO, ablations, plots, diversity
python -m pytest tests -q               # unit / smoke tests
```

The artifacts of the smoke runs shipped with this repository are in `runs_toy/`,
`figures_toy/` and `results_toy/` (see section 7).

## 6. Notes, assumptions and deviations

* **Conditioning.** The paper says the shared backbones are "conditioned on the
  parameters `phi_j`" without fixing the mechanism; the implementation
  concatenates `phi_j` to the network input (`model.phi_conditioning: concat`,
  `film` is also available). Since `phi_j` is local, gradients stay local to
  policy `j` either way.
* **Off-policy bootstrap.** Eq. 6 is written with `V_{pi_j, old}` on the
  right-hand side while the critic being regressed is `V_{pi_i}`; bootstrapping
  from the *updating* policy's old value is the consistent choice and is what is
  implemented (a 1-step TD target for `V_{pi_i}` on off-policy states), with
  `A_off` the corresponding 1-step residual.
* **Sub-sampling.** By default the leader's off-policy data set is sub-sampled
  to exactly `|D_1|` transitions, in whole sequences so that recurrent
  mini-batching (`seq_len = horizon = 16`) stays valid; the "high off-policy
  ratio" ablation uses the complete data set (Sec. 6.3).
* **Importance weights** are clamped at `algo.max_importance_weight = 10`
  (numerical safeguard, not part of the paper).
* **Evaluation.** SAPG's curves report the *leader*; DexPBT reports the best
  member of the population, since population-based training selects by
  performance (`scripts/plot_results.py` reduces population metrics with `max`).
* **Observation normalisation** is off by default (`model.normalize_obs`);
  the paper does not specify it.
* **Reward shaping weights** of the AllegroKuka tasks (reach / lift / target /
  success) are not numerically specified in the paper; `RewardScales` exposes
  them. The success definition and the tolerance curriculum follow App. A
  literally (`K = 30` hold steps, tolerance `7.5 cm -> 1 cm`, decreased by 10%
  each time the average number of successes per episode crosses 3).
* **PQL** is implemented as a parallel DDPG with a shared replay buffer and a
  logarithmic spread of exploration noise across environments ("mixed
  exploration", Sec. 5.2).

## 7. Status of the reproduction

### 7.1 What was implemented and verified

* The complete SAPG update, both aggregation schemes and all Sec. 6.3
  ablations, plus the PPO / DexPBT / PQL baselines, covered by `tests/`
  (the off-policy loss reduces to PPO on a policy's own data, the behaviour
  policy receives no gradient, clipping uses `mu`-scaled bounds, GAE / n-step
  returns match closed-form values, `phi_j` gradients stay local to their
  policy, and every trainer runs end-to-end).
* The CPU toy suite, the plotting/aggregation pipeline and both Sec. 6.4
  diversity metrics.
* The IsaacGym task definitions, reward terms and curriculum for all five
  benchmark tasks (they cannot be executed here: no GPU, IsaacGym not
  installed).

### 7.2 Measured results on the CPU surrogate

Artifacts: `runs_toy/*/progress.csv`, `figures_toy/`, `results_toy/`
(1 seed per configuration, `N = 768` environments, `M = 6` for the SAPG family,
`sigma = 0` unless stated; "successes" = landmarks captured per episode, out of
`K = 6`, so the metric is comparable with the paper's `successes` metric for the
hard tasks).

At a matched budget of 100 iterations (1.23M env steps) the ordering is exactly
the paper's Sec. 6.3 ordering - the full method is better than every variant of
it:

| variant | successes (last 10% of the run) |
| --- | --- |
| SAPG (leader/follower, `lambda=1`, subsampled) | **2.8** |
| SAPG, entropy `sigma = 0.005` | 2.3 |
| SAPG, entropy `sigma = 0.003` | 2.3 |
| SAPG, symmetric aggregation | 2.3 |
| SAPG, high off-policy ratio | 2.2 |
| SAPG, no off-policy combination | 0.24 |

`figures_toy/fig6_multimodal_collect_ablations.png` shows the corresponding
curves: removing the off-policy combination collapses performance, and both the
symmetric scheme and the high off-policy ratio are clearly worse, as reported in
Figure 6 of the paper.

The two diversity metrics reproduce the paper's trend directly:

* **MLP reconstruction error (Fig. 8 analogue)** - SAPG's visited states are
  *consistently harder to reconstruct* than PPO's for every hidden size in
  `{2,4,8,16,32,64}` (`figures_toy/fig8_diversity.png`), i.e. SAPG's state
  distribution is the less compressible / more diverse one, exactly as in
  Sec. 6.4.
* **PCA reconstruction error (Fig. 7 analogue)** - SAPG's error decreases more
  slowly with the number of principal components than PPO's
  (`figures_toy/fig7_diversity.png`).

The *headline* comparison (SAPG vs PPO) is **not** reproduced on this surrogate:
PPO reaches 3.95 successes after 250 iterations (3.07M steps) while SAPG peaks
at ~3.0 around 1M steps and then destabilises to ~1.1 by 2.5M steps, and the
table reports SAPG 1.1 vs PPO 3.9 because it averages the last 10% of each run.
The surrogate is a smoke test - a single seed, 768 environments, small MLPs, and
a bandit-like task - and it does not reproduce the paper's regime (tens of
thousands of environments, 23-DoF contact-rich manipulation, where a single
Gaussian policy provably wastes the parallel capacity). The late-run
degradation is reported rather than tuned away; it is most plausibly the
importance-weighted update (weights clamped at 10) interacting with the
adaptive learning-rate schedule, and it is the first thing to investigate with
the real GPU-scale runs.

### 7.3 Not run here

The five benchmark experiments, the full ablation grid and the batch-size study
of Figure 2 require a GPU, IsaacGym and ~50 h per run.
`scripts/run_experiments.py` writes the exact commands (5 tasks x 4 methods x 5
seeds + ablations), `scripts/run_batch_size_study.py` runs the Figure 2 sweep,
and `scripts/plot_results.py` turns the resulting logs into Figures 5/6 and
Table 1 (mean +/- standard error over seeds, as in Sec. 5.2).
