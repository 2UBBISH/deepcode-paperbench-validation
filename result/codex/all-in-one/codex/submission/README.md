# Simformer — a reproduction of "All-in-one simulation-based inference"

This repository is a from-scratch reproduction of

> M. Gloeckler, M. Deistler, C. Weilbach, F. Wood, J. H. Macke.
> *All-in-one simulation-based inference.* ICML 2024.

The **Simformer** is a probabilistic diffusion model with a transformer backbone
that is trained on the *joint* distribution ``p(theta, x)`` of the parameters and
the data of a simulator.  A single trained network can then be queried to sample
*arbitrary* conditionals of the joint distribution (posterior, likelihood,
parameter conditionals, posterior predictives, ...), it can handle missing /
unstructured data, function valued (infinite dimensional) parameters and
observation intervals (through diffusion guidance), and it can exploit known
dependency structures of the simulator through the attention mask of the
transformer.

The paper's code base (`mackelab/simformer`) was **not** used; everything here was
written from the paper text, the appendix and the provided addendum.

---

## 1. What is implemented

| Paper | Implementation | Status |
| --- | --- | --- |
| Sec. 3.1 — tokenizer for SBI (identifier, value, metadata, condition state) | [`simformer/tokenizer.py`](simformer/tokenizer.py) | ✅ |
| Sec. 3.2 — dependency modelling with attention masks, directed/undirected | [`simformer/masks.py`](simformer/masks.py) | ✅ |
| Appendix (Webb et al. 2018 graph inversion) | [`simformer/masks.py`](simformer/masks.py) `graph_inversion` | ✅ |
| Sec. 3.3 — denoising score matching with sampled condition masks ``M_C`` | [`simformer/model.py`](simformer/model.py) `compute_loss` | ✅ |
| Sec. 3.3 — sampling of arbitrary conditionals with the reverse SDE | [`simformer/model.py`](simformer/model.py) `sample` | ✅ |
| Sec. 3.4 — conditioning on intervals with diffusion guidance | [`simformer/model.py`](simformer/model.py) `Constraint` (Alg. 1, self-recurrence) | ✅ |
| Appendix A2.1 — VESDE / VPSDE, model & training hyper parameters | [`simformer/sde.py`](simformer/sde.py), [`simformer/configs.py`](simformer/configs.py) | ✅ |
| Sec. 4.1 / Fig. 4a — Gaussian Linear, Gaussian Mixture, Two Moons, SLCP vs NPE | [`simformer/tasks/benchmarks.py`](simformer/tasks/benchmarks.py), [`scripts/run_benchmark_suite.py`](scripts/run_benchmark_suite.py) | ✅ implemented, needs GPU time to train |
| Sec. 4.1 / Fig. 4b — arbitrary conditionals (incl. new tasks Tree, HMM) | [`simformer/tasks/tree_hmm.py`](simformer/tasks/tree_hmm.py), [`scripts/evaluate_c2st.py`](scripts/evaluate_c2st.py) | ✅ |
| Sec. 4.2 — Lotka-Volterra with unstructured observations | [`simformer/tasks/lotka_volterra.py`](simformer/tasks/lotka_volterra.py), [`scripts/run_lotka_volterra.py`](scripts/run_lotka_volterra.py) | ✅ |
| Sec. 4.3 — SIRD with a time dependent (functional) contact rate | [`simformer/tasks/sird.py`](simformer/tasks/sird.py), [`scripts/run_sird.py`](scripts/run_sird.py) | ✅ |
| Sec. 4.4 — Hodgkin-Huxley with energy interval constraints | [`simformer/tasks/hodgkin_huxley.py`](simformer/tasks/hodgkin_huxley.py), [`scripts/run_hodgkin_huxley.py`](scripts/run_hodgkin_huxley.py) | ✅ |
| Baselines NPE / NLE / NRE with `sbi` | [`simformer/baselines.py`](simformer/baselines.py) | ✅ |
| C2ST (random forest, 100 trees), expected coverage | [`simformer/metrics.py`](simformer/metrics.py) | ✅ |
| MCMC reference samples of arbitrary conditionals | [`simformer/reference.py`](simformer/reference.py) | ✅ |

Everything *only* described in the appendix (extended VPSDE benchmarks, NLL
evaluation, calibration figures, gravitational waves / embedding nets, guidance
benchmarks) is not part of the core contributions and was therefore not
implemented as an experiment (the utilities that are shared with the main text —
C2ST, expected coverage of the SIR task, guidance — are present).

## 2. Repository layout

```
simformer/                 # the method
  sde.py                   # VESDE / VPSDE, perturbation kernel, score targets
  tokenizer.py             # identifier / value / metadata / condition embeddings
  transformer.py           # transformer score network (+ time embedding)
  masks.py                 # base masks of all tasks, graph inversion
  problem.py               # definition of a (theta, x) inference problem
  model.py                 # Simformer: training, sampling, guidance
  configs.py               # the hyper parameters of Appendix A2.1
  tasks/                   # the simulators of the paper
    benchmarks.py          # Gaussian Linear, Gaussian Mixture, Two Moons, SLCP
    tree_hmm.py            # Tree, HMM
    lotka_volterra.py      # Sec. 4.2
    sird.py                # Sec. 4.3
    hodgkin_huxley.py      # Sec. 4.4
  reference.py             # slice sampling / MH / HMC reference conditionals
  baselines.py             # NPE / NLE / NRE with sbi
  metrics.py               # C2ST, expected coverage
  plotting.py              # figures
scripts/                   # experiment scripts (see below)
tests/test_smoke.py        # fast smoke tests of every component
```

## 3. How to run

```bash
python -m pip install -r requirements.txt

# fast sanity checks of the implementation
python tests/test_smoke.py

# short end-to-end check: Simformer posterior vs. the analytic posterior of the
# Gaussian linear task (C2ST should decrease towards 0.5)
python scripts/check_reproduction.py --n-simulations 10000 --max-epochs 20

# Fig. 4a: benchmarks, mask variants, 1k/10k/100k simulations, NPE comparison
python scripts/generate_reference_posteriors.py --mode both        # MCMC ground truth
python scripts/run_benchmark_suite.py --evaluate --baselines npe

# Sec. 4.2 (Fig. 5): Lotka-Volterra, unstructured observations
python scripts/run_lotka_volterra.py --n-simulations 100000

# Sec. 4.3 (Fig. 6): SIRD with a time dependent contact rate
python scripts/run_sird.py --n-simulations 100000

# Sec. 4.4 (Fig. 7): Hodgkin-Huxley with a guided energy constraint
python scripts/run_hodgkin_huxley.py --n-simulations 100000
```

Every script stores checkpoints, the sampled posteriors and metrics (JSON/NPZ)
under `results/` and writes the figures of the corresponding paper figure.
Long training runs are *not* executed as part of this repository; they are meant
to be launched on a GPU machine.

## 4. Method details (what the code does, and where)

### 4.1 Tokenizer (Sec. 3.1)

Every variable of ``x_hat = (theta, x)`` is represented by a token that
concatenates

1. an **identifier** embedding.  For "ordinary" variables this is a learnable
   embedding per variable; for function valued variables (the time dependent
   contact rate of the SIRD task, observations at arbitrary times) it is the sum
   of a *shared* embedding of the variable kind and a **random Fourier
   embedding** of the time point.  This is what allows the SIRD task to be
   evaluated at arbitrarily many time points with a single trained network.
2. a **value** embedding: the scalar is repeated to the desired dimensionality
   (``v -> [v, ..., v]``, as specified in the addendum).
3. an optional **metadata** embedding.
4. the **condition state**: a learnable vector embedding for conditioned
   (``True``) variables and zeros for latent (``False``) variables.

The concatenation is projected to the token dimension (50) and passed through a
small residual MLP.

### 4.2 Attention masks and graph inversion (Sec. 3.2, Appendix A1.1)

`simformer/masks.py` contains the base masks of all tasks, constructed exactly as
described in the addendum ("Task Dependencies"): the factorised Gaussian linear
mask, the dense two moons / Gaussian mixture mask, the block diagonal SLCP mask,
the tree mask, the HMM chain mask, the two metadata dependent masks of the
Lotka-Volterra and SIRD tasks and, finally, the identity/dense masks that are used
for the marginalisation experiments.

Whenever the condition state of a batch changes, the mask is adapted with **graph
inversion** (`graph_inversion`, Algorithm 1 of the addendum, Webb et al. 2018):
the graph is moralised, latent variables are eliminated with the min-fill
criterion and the required extra edges are added to the base mask.  The result is
verified in `tests/test_smoke.py`:

* Gaussian linear, posterior conditioning → exactly the 10 missing
  parameter→data edges are inserted,
* Gaussian linear, likelihood conditioning → *no* edges are inserted (as stated
  in Appendix A1.1),
* Gaussian linear, joint estimation → the moral edges are inserted,
* HMM → the sparse chain structure is preserved and the added edges are
  the ones needed to model the posterior dependencies.

For batches with very many distinct condition masks (the two random Bernoulli
masks of a training batch) the exact, sequential elimination would have to be run
for every batch element; `Simformer.build_attention_mask` therefore uses a
vectorised variant that adds the moral graph edges of the latent variables
(`moral_neighbour_mask`) above a configurable number of distinct masks
(`max_unique_masks`).  The exact algorithm is used for the joint, posterior and
likelihood masks, i.e. for all structurally relevant cases.

### 4.3 Training (Sec. 3.3, Appendix A2.1)

* ``x_hat_t^{M_C} = (1 - M_C) * x_hat_t + M_C * x_hat_0``: conditioned variables
  stay clean.
* loss ``|| (1 - M_C) * (s_phi(x_hat_t^{M_C}, t) - grad log p_t(x_t | x_0)) ||^2``
  with the analytic denoising target ``-eps / sigma(t)`` and the variance
  weighting ``lambda(t) = sigma(t)^2``.
* ``M_C`` is sampled for every element of the batch, uniformly at random from
  {joint mask, posterior mask, likelihood mask, ``Ber(0.3)``, ``Ber(0.7)``}
  (`condition_mask_options="all"`); `"posterior"` restricts training to the
  posterior mask ("Simformer (posterior only)" of Appendix A3.1).
* diffusion times ``t ~ U[1e-5, 1]``, VESDE (``sigma_max = 15``,
  ``sigma_min = 1e-4``) by default and VPSDE (``beta_min = 0.01``,
  ``beta_max = 10``) as an option,
* Adam, batch size 1000, learning rate ``1e-4``, early stopping on the validation
  loss (10% of the simulations are held out).
* the transformer has a token dimension of 50, 6 layers (8 layers for the
  Lotka-Volterra / SIRD / Hodgkin-Huxley tasks), 4 heads, attention size 10,
  widening factor 3 and a 128-dimensional random Gaussian Fourier embedding of
  the diffusion time whose linear projection is added to the output of *each*
  feed-forward block.

### 4.4 Sampling and guidance (Sec. 3.3/3.4, Algorithm 1)

`Simformer.sample` draws from the terminal distribution, runs the reverse
SDE with Euler–Maruyama (500 steps by default) on the latent variables only and
keeps the conditioned variables fixed.  Interval or general constraints
``c(x_hat) <= 0`` are enforced with diffusion guidance

```
s(x_t, t | c) ~ s_phi(x_t, t) + grad_{x_t} log sigmoid(-s(t) c(x_hat_0)),
x_hat_0 = (x_t + sigma(t)^2 s_phi(x_t, t)) / mu(t)
```

with the scaling function ``s(t) = 1 / sigma(t)^2`` and optional self-recurrence
(``self_recurrence``/``r``).  `Interval`, `IntervalUpperBound`,
`IntervalLowerBound`, `LinearConstraint` and `CallableConstraint` are provided.

## 5. Experiments and what was actually verified in this environment

The compute budget for this reproduction was a CPU-only machine and a few hours
of wall clock time.  Therefore the paper-scale training runs (1e5 simulations,
6-8 layers, 500 diffusion steps, 100k training iterations) could not be executed
here; the code and the README are written so that those runs can be launched
directly (see Sec. 3).  What *was* executed:

| Check | Result |
| --- | --- |
| `tests/test_smoke.py` (10 tests: tokenizer, SDE marginal/score identities, transformer shapes/masks, graph inversion properties, sampling, guidance, Fourier identifiers) | all pass |
| `scripts/check_reproduction.py` (Gaussian linear, 10k simulations, 20 epochs, 6 layers) | C2ST vs. the analytic posterior decreases 1.000 → 0.9885 → 0.9855 → 0.982 monotonically with training (perfect = 0.5); the raw output is committed in [`reproduction_evidence/reproduction_check.json`](reproduction_evidence/reproduction_check.json) |
| CLI pipeline end to end on a tiny budget (`train_simformer.py` → `generate_reference_posteriors.py` → `evaluate_c2st.py`) | runs, writes checkpoints, reference samples and `c2st.json` (posterior **and** arbitrary conditional C2ST) |
| Lotka-Volterra simulator, mask, structured/unstructured observations, log joint | runs; ODE + Gaussian noise as specified |
| SIRD simulator, GP prior, log-normal noise, time dependent mask, switching to a finer inference grid | runs |
| Hodgkin-Huxley simulator (200 ms, dt = 0.01 ms, 4 µA/cm² between 50 and 150 ms, 0.05 dW_t) | runs; spike counts 0-21, resting potentials ≈ -65 mV, sodium based energy 0.014-0.078 µJ/s |
| MCMC reference samplers (slice sampling, MH, HMC) | implemented after Appendix A2.2 |
| `sbi` baselines NPE / NLE | trained and sampled on tiny budgets (two moons, SLCP): the NPE posterior mean recovers the ground truth parameter, the NLE likelihood sampler returns finite samples; NRE uses the same code path with `sample_with="mcmc"` |

### Deviations and assumptions (explicitly flagged)

* **Hodgkin-Huxley priors.** The paper refers to Pospischil et al. (2008) but
  does not restate the prior ranges.  We use uniform priors centred on the
  parameters of their Hodgkin-Huxley model
  (`g_Na ∈ [10, 120]`, `g_K ∈ [3, 20]`, `g_L ∈ [0.005, 0.05]`, `E_Na ∈ [40, 60]`,
  `E_K ∈ [-95, -75]`, `E_L ∈ [-80, -60]`, `C_m ∈ [0.5, 2]`); with these priors the
  simulator produces between 0 and ~20 spikes in the 100 ms injection window.
* **Lotka-Volterra prior.** Implemented as stated in the paper: a
  sigmoid-transformed Normal, scaled to ``[1, 3]`` (``theta = 1 + 2 sigmoid(raw)``,
  ``raw ~ N(0, 1)``); initial state ``(1, 1)``; the posterior is inferred on a
  uniform grid between ``t = 0`` and ``t = 15`` with 16 grid points and Gaussian
  observation noise ``sigma = 0.1``.
* **Two moons noise.** The paper writes ``r ~ N(0.1, 0.012)``; we use a normal
  with mean 0.1 and standard deviation 0.012.
* **SIRD initial condition.** Not stated in the paper; we use the fractions
  ``(S, I, R, D) = (0.99, 0.01, 0, 0)``, a time window of ``[0, 60]`` and a
  log-normal observation model whose mean equals the simulated value.
* **Number of variables per task.** The addendum lists generic mask templates
  (e.g. a ``10 x 10`` mask for two moons); we build the same *structure* with the
  number of variables of the actual task (two moons: 2 + 2, tree: 3 + 4).
* **Exact vs. vectorised graph inversion for random condition masks** — see
  Sec. 4.2 above.
* **C2ST evaluation.** We use the random forest with 100 trees as specified and
  average over 5-fold cross validation to reduce the variance of the estimate;
  the paper reports a single split.

## 6. Relation to the paper's figures

| Figure | Script | What it produces |
| --- | --- | --- |
| Fig. 1 / 2 | — | conceptual; the architecture is implemented in `simformer/` |
| Fig. 3 | `scripts/evaluate_c2st.py` (conditional mode) | samples of arbitrary conditionals; the paper's example is the two moons task |
| Fig. 4a | `scripts/run_benchmark_suite.py --evaluate` | C2ST of the posterior for 4 tasks × 3 mask variants × {1k, 10k, 100k} simulations, incl. NPE |
| Fig. 4b | `scripts/generate_reference_posteriors.py --mode conditional` + `scripts/evaluate_c2st.py` | C2ST of arbitrary conditionals (100 randomly selected conditionals per task) |
| Fig. 5 | `scripts/run_lotka_volterra.py` | posterior / posterior predictive for 4 prey and 4 prey + 9 predator observations, C2ST against MCMC |
| Fig. 6 | `scripts/run_sird.py` | posterior of the global parameters and of ``beta(t)``; expected coverage |
| Fig. 7 | `scripts/run_hodgkin_huxley.py` | posterior marginals, posterior predictive energy, guided energy constrained posterior |
