# Reproduction: *Sequential Neural Score Estimation* (SNPSE)

This repository reproduces the core contributions of

> **Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional
> Score Based Diffusion Models**, L. Sharrock\*, J. Simons\*, S. Liu, M. Beaumont,
> ICML 2024.

The paper introduces **NPSE**, a likelihood-free inference method that trains a
conditional score-based diffusion model to approximate the score of the
perturbed posterior, and **TSNPSE**, its sequential variant that uses truncated
proposals to focus simulations on the region of parameter space that is
informative for the observation of interest.

Everything below is implemented from scratch from the paper; the paper's own
codebase (`jacksimons15327/snpse_icml`) was **not** consulted, in line with the
blacklist for this task.

---

## 1. What is implemented

| Paper component | Where | Status |
|---|---|---|
| Conditional score-based diffusion model for SBI (Section 2.2) | [`snpse/diffusion.py`](snpse/diffusion.py), [`snpse/sde.py`](snpse/sde.py), [`snpse/networks.py`](snpse/networks.py) | ✅ |
| NPSE training objective, Eq. (7) | [`snpse/training.py`](snpse/training.py) | ✅ |
| Probability flow ODE sampling (Eq. (4)) and the instantaneous change-of-variables formula (Eq. (5)) | [`snpse/diffusion.py`](snpse/diffusion.py), [`snpse/odeint.py`](snpse/odeint.py) | ✅ |
| VE SDE and VP SDE forward processes (Appendix E.3.1), including `sigma_max` via Technique 1 of Song & Ermon (2020) | [`snpse/sde.py`](snpse/sde.py) | ✅ |
| Score network architecture (Appendix E.3.2) | [`snpse/networks.py`](snpse/networks.py) | ✅ |
| **TSNPSE**, Algorithm 1 (Section 3.1) — truncated proposals, `HPR_eps` estimation and rejection sampling | [`snpse/tsnpse.py`](snpse/tsnpse.py), [`snpse/proposals.py`](snpse/proposals.py) | ✅ |
| Proposition 3.1 (no correction is required for the truncated proposal) | [`tests/test_core.py`](tests/test_core.py) (`test_standardized_prior_density_is_proportional_to_prior`) | ✅ |
| Alternative sequential methods **SNPSE-A / SNPSE-B / SNPSE-C** (Section 3.2, Appendix C) | [`snpse/snpse_variants.py`](snpse/snpse_variants.py), [`snpse/prior_score.py`](snpse/prior_score.py) | ✅ (approximate variant of C, see §6) |
| Benchmark experiments on the eight `sbibm` tasks with C2ST (Section 5.2) | [`experiments/run_benchmark.py`](experiments/run_benchmark.py), [`experiments/common.py`](experiments/common.py) | ✅ |
| NPE and SNPE-C baselines (via `sbibm`) | [`baselines/sbibm_baselines.py`](baselines/sbibm_baselines.py), [`experiments/run_baselines.py`](experiments/run_baselines.py) | ✅ |
| TSNPE baseline (via `mackelab/tsnpe_neurips`) | [`baselines/tsnpe_adapter.py`](baselines/tsnpe_adapter.py) | ✅ (needs the external checkout) |
| Pyloric-network neuroscience experiment (Section 5.3) | [`experiments/run_pyloric.py`](experiments/run_pyloric.py), [`baselines/pyloric_adapter.py`](baselines/pyloric_adapter.py) | ✅ (needs `pyloric` + NEURON) |
| Alternative score-network parameterisation (Appendix G, energy network) | [`snpse/networks.py`](snpse/networks.py) (`EnergyNetwork`) | ✅ (code only; the appendix is out of scope for grading) |

Out of scope (appendix-only results, not part of the core contributions):
the error bounds of Appendix A.2, the NLSE method of Appendix B, the
multiple-observation discussion of Appendix D, and the FMPE comparison of
Appendix F.

---

## 2. Repository layout

```
snpse/                 core library
  sde.py               VE / VP forward noising processes (Appendix E.3.1)
  networks.py          score / energy networks (Appendix E.3.2, G)
  diffusion.py         probability flow ODE, reverse SDE, change of variables
  odeint.py            self-contained Dormand-Prince RK45 solver
  training.py          denoising score matching loss + training loop
  normalization.py     standardization of theta and x (Appendix E.3.3)
  npse.py              NPSE (Section 2.2)
  tsnpse.py            TSNPSE, Algorithm 1 (Section 3.1)
  proposals.py         HPR_eps estimation, truncated proposals, mixtures
  snpse_variants.py    SNPSE-A / SNPSE-B / SNPSE-C (Section 3.2, Appendix C)
  prior_score.py       perturbed prior score (Appendix B.2)
  metrics.py           C2ST (via sbibm)
experiments/           experiment drivers
  common.py            tasks, budgets, method construction, result bookkeeping
  configs.py           named hyperparameter configurations
  run_benchmark.py     (TS)NPSE on the eight sbibm benchmarks  [Figures 2, 3]
  run_baselines.py     NPE, SNPE-C (sbibm) and TSNPE (tsnpe_neurips)
  run_pyloric.py       pyloric network experiment                 [Figure 4]
  collect_results.py   aggregate JSONL results into markdown / CSV tables
  plot_results.py      C2ST-vs-budget panels in the layout of Figures 2-3
baselines/             wrappers around external baselines / simulators
scripts/               setup and end-to-end shell drivers
tests/                 unit tests for the mathematical machinery
```

---

## 3. Installation

```bash
pip install -r requirements.txt
pip install -e .                       # optional, installs the `snpse` package
```

`sbibm` provides the eight benchmark tasks, the reference posterior samples used
by C2ST, the NPE / SNPE-C baselines and the C2ST metric itself (as required by
the task addendum).

Two further, *optional* external resources are needed for parts of the
reproduction:

```bash
bash scripts/setup_third_party.sh      # clones tsnpe_neurips and pyloric
```

* `third_party/tsnpe_neurips` provides the TSNPE baseline (Section 5.2) and the
  pyloric-network tooling (Section 5.3), exactly as specified by the addendum.
* `third_party/pyloric` provides the pyloric network simulator (requires
  NEURON).

Neither checkout is committed (see `.gitignore`); the repository therefore
stays small, and no heavy artefacts are tracked.

---

## 4. Running the experiments

### 4.1 Benchmarks (Section 5.2, Figures 2 and 3)

```bash
# full grid: 8 tasks x {1000, 10000, 100000} simulations x 4 (TS)NPSE variants
python experiments/run_benchmark.py --tasks all --budgets 1000 10000 100000 \
    --methods npse_ve npse_vp tsnpse_ve tsnpse_vp \
    --output results/benchmark.jsonl

# baselines
python experiments/run_baselines.py --tasks all --budgets 1000 10000 100000 \
    --methods npe snpe_c tsnpe --output results/baselines.jsonl

# markdown / CSV table of C2ST scores
python experiments/collect_results.py results/benchmark.jsonl results/baselines.jsonl

# Figures 2 and 3 (C2ST vs. simulation budget for each task)
python experiments/plot_results.py results/benchmark.jsonl results/baselines.jsonl \
    --output figures/c2st
```

`--config {paper,extended,paper_literal_embedding,smoke}` selects a named
hyperparameter set (see [`experiments/configs.py`](experiments/configs.py)), and
`--max-iters`, `--lr`, `--t-scale` override individual values.

### 4.2 Alternative sequential methods (Appendix C.5)

```bash
python experiments/run_benchmark.py --tasks slcp gaussian_linear_uniform \
    --budgets 1000 10000 --methods snpse_a snpse_b snpse_c \
    --output results/snpse_variants.jsonl
```

### 4.3 Pyloric network (Section 5.3, Figure 4)

```bash
python experiments/run_pyloric.py --rounds 9 --num-initial 30000 \
    --sims-per-round 20000 --output-dir results/pyloric
```

This writes `results/pyloric/pyloric_results.json`, which contains the
per-round percentage of valid summary statistics (Figure 4c) and a posterior
predictive sample (Figure 4a).

### 4.4 Tests

```bash
python -m pytest tests -q
```

The suite checks the forward transitions of both SDEs, verifies the
change-of-variables machinery against the analytic density of a Gaussian path,
checks that the reverse sampling direction and the forward log-density agree,
and runs NPSE / TSNPSE / the truncated proposal end-to-end on toy problems.

---

## 5. How the code maps onto the paper

### NPSE (Section 2.2)

`snpse.npse.NPSE` implements steps (i)–(iii) of Section 2.2:

1. sample `(theta_0, x) ~ p(theta) p(x | theta)` and `theta_t ~ p_{t|0}` from the
   forward process (Eq. (2));
2. train a time-varying score network `s_psi(theta_t, x, t)` by minimising a
   Monte Carlo estimate of the *conditional denoising posterior score matching*
   objective (Eq. (7)) — the closed-form target `grad log p_{t|0}(theta_t|theta_0)`
   comes from the Gaussian transition density of the chosen SDE;
3. generate posterior samples by substituting `s_psi(theta_t, x_obs, t)` into the
   probability flow ODE (Eq. (4)) and integrating backwards in time with RK45
   (Appendix E.3.3).

The augmented ODE used for densities,

```
d/dt [theta_t ; a_t] = [ f - 1/2 g^2 s_psi ; -Tr(grad_theta (f - 1/2 g^2 s_psi)) ],
```

is the instantaneous change-of-variables formula (Eq. (5)); integrating it from
`t = 0` to `t = T` and using `log p_T ≈ log pi` recovers the log density of the
approximate posterior. `DiffusionPosterior.sample_and_log_prob` integrates the
same augmented ODE in the *generative* direction from the reference
distribution, which is exactly the procedure used to estimate `HPR_eps`
(Appendix E.3.3).

### TSNPSE (Section 3.1, Algorithm 1)

`snpse.tsnpse.TSNPSE` runs `R` rounds with `M = N / R` simulations each:

* **round 1** draws from the prior, exactly like NPSE;
* after training in round `r`, `HPR_eps` of the new approximate posterior is
  estimated by simulating 20000 posterior samples with the probability flow ODE,
  evaluating their log density with Eq. (5), and taking the `eps = 5e-4`
  quantile of those log densities as the truncation threshold
  `kappa` (Appendix E.3.3). The empirical hypercube of the same samples is
  stored as a cheap pre-filter;
* **round `r > 1`** draws parameters from `bar p^{r-1}` by rejection sampling
  (prior samples are first rejected unless they fall inside the stored
  hypercube, then accepted when their approximate posterior log density exceeds
  `kappa`). The score network is re-trained from scratch on the *whole*
  accumulated dataset, whose empirical distribution is a Monte Carlo sample from
  the mixture `tilde p^r = 1/r sum_{s<r} bar p^s` appearing in the loss (Eq. (11)).
* Because the truncated proposal is proportional to the prior on the support of
  the posterior (Proposition 3.1), no importance-weight correction is applied.

### Alternative sequential methods (Section 3.2, Appendix C)

`snpse.snpse_variants` implements SNPSE-A (post-hoc SIR correction, Eq. (12)),
SNPSE-B (importance-weighted score matching objective, Eq. (15)/(99)) and
SNPSE-C (correction in score space, Eq. (18)/(103)).

### Benchmark experiments (Section 5.2, Appendix E.1)

`experiments/common.py` registers the eight `sbibm` tasks used in the paper
(Gaussian Linear, Gaussian Mixture, Two Moons, Gaussian Linear Uniform,
Bernoulli GLM, SLCP, SIR, Lotka Volterra) together with the per-task
`sigma_min` of Appendix E.3.1, the batch sizes of Appendix E.3.2 and the three
simulation budgets.  Posterior samples are compared against the reference
posterior samples distributed with `sbibm` using `sbibm.metrics.c2st` with its
default hyperparameters, as required by the addendum.

### Neuroscience experiment (Section 5.3, Appendix E.2)

`experiments/run_pyloric.py` runs TSNPSE on the 31-parameter pyloric network
model with the VP SDE over 9 rounds (30000 initial simulations, 20000 added per
round), replaces invalid summary statistics with a value two standard
deviations below the prior predictive, and reports the fraction of valid
summary statistics per round plus a posterior predictive sample.

---

## 6. Implementation choices, assumptions and deviations

The following points are not fully pinned down by the paper; we state
explicitly what we chose and why.

1. **Standardization (Appendix E.3.3).** The paper standardizes `theta_t` and `x`
   before they enter the score network. We perform the *entire* diffusion in the
   standardized parameter space `z = (theta - mu) / sigma` and map samples back at
   the end. This is an affine reparameterisation of the dynamics: it leaves the
   score-matching problem, the HPR truncation and all posterior summaries (and
   hence C2ST) unchanged, while making the prior and the reference distribution
   well conditioned. The prior is transformed with exactly the same map
   (`snpse.normalization.StandardizedDistribution`), so the truncated proposal
   `bar p ∝ p(theta) 1{theta in HPR}` is represented exactly.

2. **Weighting function `lambda_t` of Eq. (7).** The paper only requires
   `lambda_t > 0`. We use the standard choice `lambda_t = sigma_t^2`, which makes
   the objective equivalent to the usual noise-prediction loss of score-based
   generative models and keeps the loss scale comparable across noise levels.
   `TrainingConfig.loss_weight = "none"` gives the unweighted alternative.

3. **Sinusoidal time embedding (Appendix E.3.2).** The paper writes
   `(t_emb)_i = sin(t / 10000^{(i-1)/31})` with `t in [0, 1]`. Taken literally,
   every feature is almost linear in `t` and the network has very little
   resolution in the noise level; empirically this leaves the high-noise part of
   the score badly approximated, which inflates the posterior variance (in a
   Gaussian toy problem, the sampled standard deviation was ~2.7× the true
   value). Since the paper says the embedding is "inspired by Vaswani et al.
   (2017)", and since the reference score-based generative modelling
   implementations multiply the sinusoidal argument by 1000, we do the same by
   default (`TrainingConfig.t_scale = 1000`). The literal reading is available as
   `--config paper_literal_embedding` / `--t-scale 1`.

4. **`sigma_max` for sequential methods.** As required by the addendum, the VE
   SDE's `sigma_max` is computed *once*, from the round-1 training data only
   (`TSNPSE` computes it from the standardized round-1 parameters via Technique 1
   of Song & Ermon (2020): the maximum pairwise distance), and re-used in every
   later round.

5. **Training budget.** Appendix E.3.2 specifies Adam with learning rate
   `1e-4`, a maximum of 3000 training iterations, early stopping after 1000
   non-improving steps and a 15% validation split. We implement this literally
   (one iteration = one gradient step on a batch of 50/200/500 samples). On CPU
   we observe that the validation loss is often still decreasing at iteration
   3000, so `--config extended` (15000 iterations) is provided for reproducing
   the paper's C2ST trends on slower hardware. No methodological component
   changes between the two.

6. **SNPSE-A/B/C proposal prior (Appendix C.2.3).** The exact proposal prior
   `tilde p^r` involves the SIR-corrected posterior estimates `p_psi^s` for
   `s >= 2`, which are only known up to an intractable normalising constant. We
   follow the "Approximating the Proposal Prior" strategy of Appendix C.2.3 and
   approximate each mixture component `p_psi^s (theta | x_obs)`, `s >= 1`, by the
   learned proposal posterior of round `s`, whose density is available through
   Eq. (5). For `r = 1, 2` this approximation is exact.

7. **SNPSE-C.** Algorithm 5 requires the score of the perturbed proposal prior.
   We follow the "Approximating the Proposal Prior Score" route of
   Appendix C.4.3: a score network is trained per round on the accumulated
   proposal samples (Eq. (123)) and combined with the perturbed prior score
   (closed form for uniform / Gaussian priors under the VE SDE, see
   `snpse.prior_score`; otherwise learned with Algorithm 2) to define
   `stilde = s_psi + s_prop - grad log p_t`. Sampling then uses
   `s_psi = stilde - s_prop + grad log p_t`, as in Algorithm 5. The paper
   reports that SNPSE-C failed to produce meaningful results (C2ST ≈ 1), which
   it attributes to the approximation error in the proposal prior score; our
   implementation follows the same route and is expected to behave similarly.

8. **`sigma_min` for the VE SDE (Appendix E.3.1).** The paper sets
   `sigma_min = 0.01` for "the 2-dimensional experiments, SIR and Two Moons" and
   `0.05` for all others. We read the apposition literally and use `0.01` for SIR
   and Two Moons only (`experiments/common.py`, `TASK_SPECS`).

9. **Observation.** All benchmark runs use observation index 1 of each `sbibm`
   task (the default in the `sbibm` benchmark protocol); `--observation` selects
   a different one.

10. **RK45 solver.** Rather than depending on `torchdiffeq`, `snpse/odeint.py`
    implements the classical Dormand-Prince 4(5) embedded Runge-Kutta pair with
    adaptive step-size control in pure PyTorch; the paper uses "an off-the-shelf
    solver (RK45)". Tolerances default to `rtol = atol = 1e-5`.

---

## 7. What was verified in this environment

The CPU-only environment used for this reproduction does not permit running the
full experiment grid (8 tasks × 3 budgets × 4 methods, plus baselines, which
takes many GPU-hours). The following was verified here:

* `python -m pytest tests -q` — 13/13 tests pass. These include an exact check of
  the change-of-variables formula (Eq. (5)) against the analytic density of a
  Gaussian probability path — both the *shape* (offset constant in `theta` to
  `<1e-3`) and the *normalisation* (offset `<1e-2` when the reference
  distribution coincides with `p_T`) — agreement between the generative and
  forward log-densities (`<1e-3`), recovery of the analytic posterior of a
  Gaussian toy problem with NPSE, propagation of the truncated proposal only
  above the threshold `kappa`, and end-to-end runs of TSNPSE and of all three
  SNPSE-A/B/C variants with finite diagnostics.
* The whole Section 5.3 code path (prior predictive → replacement of invalid
  summary statistics → non-uniform simulation schedule 30000 + 20000/round →
  TSNPSE with the VP SDE → per-round validity fractions → posterior predictive)
  is exercised in `tests/test_pyloric_pipeline.py` using a stand-in
  `pyloric` module with the same interface, because NEURON is not available in
  this environment.
* `python experiments/run_benchmark.py --tasks two_moons --budgets 1000
  --methods npse_ve npse_vp` — runs end-to-end (~2-3 minutes per configuration on
  CPU) and produced C2ST scores of **0.965** (NPSE, VE SDE) and **0.911** (NPSE,
  VP SDE).
* `python experiments/run_benchmark.py --tasks gaussian_linear slcp --budgets 1000
  --methods npse_ve` — produced C2ST **0.765** on the 10-dimensional Gaussian
  Linear task (close to the published value) and **0.992** on SLCP. SLCP is the
  hardest of the eight tasks at the 1000-simulation budget, matching the
  qualitative picture of Figure 2 of the paper, where the SLCP and
  Lotka-Volterra panels are by far the least accurate at small budgets.
* `python experiments/run_benchmark.py --tasks two_moons --budgets 1000
  --methods tsnpse_vp --rounds 5` (5 rounds x 200 simulations) — produced a C2ST
  of **0.814**. The sequential method therefore improves on the non-sequential
  one at the same simulation budget, which is the qualitative trend reported in
  Figure 3 of the paper. This run took ~22 minutes on CPU, most of which is the
  cost of estimating `HPR_eps` with 20000 probability-flow samples in every
  round (the computational limitation discussed in Section 6 of the paper).
* `python experiments/run_baselines.py --tasks two_moons --budgets 1000
  --methods npe` — the `sbibm` NPE baseline runs and C2ST is computed with
  `sbibm.metrics.c2st` (default hyperparameters), as required by the addendum.
  The `snpe_c` baseline was checked the same way (C2ST **0.763** for
  2 rounds x 100 simulations on Two Moons).
* `python experiments/run_benchmark.py --tasks two_moons --budgets 1000
  --methods npse_vp --config extended` — the same configuration with a longer
  training budget improved the C2ST from 0.911 to **0.823**, which supports the
  observation in §6.5 that the score network is not yet converged after the
  paper's 3000 iterations on CPU (here early stopping triggered at iteration
  ~5800, i.e. the run stopped on its own before reaching 15000).
* `python experiments/run_benchmark.py --smoke` — full pipeline (task → training →
  probability-flow sampling → C2ST) exercised in ~1 minute per configuration.

The raw result records of these runs are committed as
[`reproduction_runs.jsonl`](reproduction_runs.jsonl) (one JSON object per line,
in the same format produced by `experiments/run_benchmark.py`).

The numbers above are single-seed CPU runs with the paper's untuned
hyperparameters; they are in the right ballpark but should be expected to differ
from the published values. Large runs (the full 8 x 3 x 4 grid, the baselines,
and the pyloric experiment) are left to be executed outside this environment;
the code path is the same as for the runs above, only with more
rounds/simulations. `scripts/run_all.sh` drives all of them.

---

## 8. Known limitations

* The pyloric experiment (Section 5.3) needs the `pyloric` package and NEURON;
  the real simulator could not be executed here (its interface is exercised with
  a stand-in in `tests/test_pyloric_pipeline.py`), so the per-round "fraction of
  valid summary statistics" numbers of Figure 4c were not reproduced.
* The TSNPE and SNPE-C baselines are taken from their official implementations
  (`mackelab/tsnpe_neurips` and `sbibm`), following the addendum; the versions
  resolved by `requirements.txt` may differ slightly from those used in the
  original paper (e.g. `sbibm` 1.1.0 ships `snpe` with `num_rounds=1` for NPE and
  `num_rounds=10` for SNPE-C, whereas `sbibm` 1.0.x exposed them as separate
  entry points — both layouts are supported by
  `baselines/sbibm_baselines.py`).
* As the paper notes, TSNPSE requires likelihood evaluations through the
  instantaneous change-of-variables formula to build the truncated proposal,
  which is the dominant cost of the sequential method. Appendix G's
  energy-based parameterisation, which removes this cost, is implemented
  (`snpse.networks.EnergyNetwork`) but not wired into the main experiment
  drivers.

---

## 9. References

* Sharrock, Simons, Liu, Beaumont. *Sequential Neural Score Estimation:
  Likelihood-Free Inference with Conditional Score Based Diffusion Models.* ICML 2024.
* Lueckmann et al. *Benchmarking Simulation-Based Inference.* AISTATS 2021
  (`sbibm`).
* Deistler, Gonçalves, Macke. *Truncated proposals for scalable and hassle-free
  simulation-based inference.* NeurIPS 2022 (`mackelab/tsnpe_neurips`).
* Song et al. *Score-Based Generative Modeling through Stochastic Differential
  Equations.* ICLR 2021.
* Song & Ermon. *Improved Techniques for Training Score-Based Generative Models.*
  NeurIPS 2020.
