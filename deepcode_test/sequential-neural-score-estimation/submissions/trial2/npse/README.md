# Neural Posterior Score Estimation (NPSE)

Reference implementation for the paper:

> **Neural Posterior Score Estimation for Simulation-Based Inference**

This repository reproduces:

- **NPSE** — non-sequential neural posterior score estimation using time-varying
  score-based diffusion models.
- **TSNPSE** — truncated sequential neural posterior score estimation.
- **SNPSE-A / SNPSE-B / SNPSE-C** — sequential NPSE variants.
- **NLSE** — neural likelihood score estimation baseline.
- Benchmark experiments on eight SBI tasks and the pyloric-network application.

The code follows a score-based diffusion formulation. We define a forward SDE
that perturbs simulator parameters, train a conditional score network with
denoising score matching, and sample approximate posteriors by solving the
reverse-time probability-flow ODE.

---

## Installation

Python 3.9 or 3.10 is recommended.

```bash
cd npse
pip install -r requirements.txt
```

Core dependencies:

- `torch >= 1.13`
- `torchdiffeq >= 0.2.3` (probability-flow ODE integration and density evaluation)
- `sbibm >= 1.0.0` (benchmark simulators, reference posterior samples, baselines)
- `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`, `pyyaml`, `tqdm`
- `pytest` (unit validation)

A GPU is recommended, especially for simulation budgets of 100 000.
CPU is feasible for low-dimensional benchmarks and small budgets, although
ODE-based density evaluation (used by TSNPSE/SNPSE) is expensive.

> `sbibm` is used primarily for reference posterior samples and stored
> observations. Every benchmark also ships a self-contained fallback simulator,
> so training and sampling can proceed even if `sbibm` is unavailable.

---

## Project layout

```
npse/
├── README.md
├── requirements.txt
├── config/
│   ├── default.yaml          # global hyperparameters, SDE choices, budgets
│   └── benchmarks.yaml       # benchmark-specific settings
├── src/
│   ├── sde.py                # VE/VP SDE definitions and transition kernels
│   ├── networks.py           # theta/x embeddings, time embedding, score nets
│   ├── losses.py             # DSM losses for NPSE, NLSE, prior, SNPSE variants
│   ├── sampling.py           # probability-flow ODE sampler
│   ├── density.py            # instantaneous change-of-variables density
│   ├── npse.py               # non-sequential NPSE trainer
│   ├── tsnpse.py             # truncated sequential TSNPSE trainer
│   ├── snpse.py              # SNPSE-A, SNPSE-B, SNPSE-C
│   ├── nlse.py               # Neural Likelihood Score Estimation
│   ├── prior.py              # exact/estimated perturbed prior scores
│   ├── hpr.py                # HPR truncation and truncated proposal sampler
│   └── metrics.py            # C2ST, MMD, coverage, posterior predictive checks
├── benchmarks/
│   ├── base.py
│   ├── gaussian_linear.py
│   ├── gaussian_mixture.py
│   ├── two_moons.py
│   ├── gaussian_linear_uniform.py
│   ├── bernoulli_glm.py
│   ├── slcp.py
│   ├── sir.py
│   └── lotka_volterra.py
├── experiments/
│   ├── train_npse_benchmarks.py
│   ├── train_tsnpse_benchmarks.py
│   ├── run_snpse_variants.py
│   ├── run_nlse_comparison.py
│   ├── run_pyloric.py
│   ├── evaluate_benchmarks.py
│   └── make_figures.py
└── tests/
    ├── test_sde.py
    ├── test_networks.py
    ├── test_losses.py
    ├── test_sampling.py
    └── test_benchmarks.py
```

---

## SDE choices

Two score-based diffusion SDEs are implemented in `src/sde.py`.

### Variance Exploding (VE)

```
dθ_t = σ_min * (σ_min / σ_max)^t * sqrt(2 * log(σ_max / σ_min)) dW_t
```

Transition:

```
p_{t|0}(θ_t | θ_0) = N(θ_t | θ_0, σ_min² * (σ_max / σ_min)^(2t) I)
```

Target denoising score:

```
∇_θ log p_{t|0}(θ_t | θ_0) = -(θ_t - θ_0) / variance
```

- `σ_min = 0.01` for two-dimensional tasks, SIR, and Two Moons.
- `σ_min = 0.05` otherwise.
- `σ_max` is calibrated with Technique 1 of Song & Ermon (2020)
  (maximum pairwise Euclidean distance among prior samples).

### Variance Preserving (VP)

```
dθ_t = -0.5 β_t θ_t dt + sqrt(β_t) dW_t
β_t  = β_min + t * (β_max - β_min),  β_min = 0.1, β_max = 11.0
```

Transition:

```
μ_t = exp(-0.5 * ∫_0^t β_s ds) * θ_0
Σ_t = (1 - exp(-∫_0^t β_s ds)) I
```

Target denoising score:

```
∇_θ log p_{t|0}(θ_t | θ_0) = -(θ_t - μ_t) / Σ_t
```

### Reverse dynamics

Reverse SDE:

```
dθ_t = [f(θ_t, t) - g(t)² ∇_θ log p_t(θ_t|x)] dt + g(t) dW̄_t
```

Probability-flow ODE (used for sampling and density evaluation):

```
dθ_t = [f(θ_t, t) - 0.5 g(t)² ∇_θ log p_t(θ_t|x)] dt
```

All sampling solves the probability-flow ODE backward from `t=T` to `t=0`
with RK45 (default `rtol=1e-5`, `atol=1e-5`).

---

## Score network architecture

Defined in `src/networks.py`:

- Theta embedding: 3-layer MLP, 256 hidden units, output `max(30, 4*d)`.
- Observation embedding: 3-layer MLP, 256 hidden units, output `max(30, 4*p)`.
- Sinusoidal time embedding: 64 dimensions (32 sine + 32 cosine terms).
- Final score MLP: concatenates `[theta_emb, x_emb, t_emb]`, 3 layers,
  256 hidden units, output dimension `d`.
- All MLPs use SiLU activations.
- Both `θ_t` and `x` are standardized per dimension before entering the network.

---

## Methods

### NPSE (non-sequential)

For each training sample:

```
θ_0 ~ p(θ),  x ~ p(x|θ_0),  t ~ U(0, T),  θ_t ~ p_{t|0}(θ_t|θ_0)
```

Objective:

```
J_NPSE_DSM(ψ) = 1/2 ∫ λ_t E[ ||s_ψ(θ_t, x, t) - ∇_θ log p_{t|0}(θ_t|θ_0)||² ] dt
```

The default weighting is `λ_t = g(t)²`. At the optimum, the network equals the
posterior score `∇_θ log p_t(θ_t|x)` (Appendix A.1 of the paper).

### TSNPSE (truncated sequential)

Sequential rounds `R=10` by default, with per-round budget `M = N/R`.
The round-`r` proposal prior is the running truncated mixture:

```
p̃^r(θ) ∝ c^r(θ) p(θ),   c^r(θ) = (1/r) Σ_{s=0}^{r-1} 1{θ ∈ Θ^s}
```

where `Θ^0` is the prior support and for `s ≥ 1`:

```
Θ^s = HPR_ε(p_ψ^s(θ | x_obs))
```

HPR truncation:

- `ε = 5e-4`.
- Draw 20 000 approximate posterior samples.
- Evaluate their log densities with instantaneous change-of-variables.
- Set the threshold `κ` to the `ε`-quantile of log densities.
- Rejection-sample from the prior with a cheap hypercube pre-rejection using
  posterior-sample min/max bounds, then accept if `log density > κ`.

All rounds' samples are accumulated and the score network is retrained from
scratch on the combined dataset.

### SNPSE variants

- **SNPSE-A**: learn the proposal posterior score, sample with the
  probability-flow ODE, then apply sampling-importance-resampling with
  `h_i = p(θ_i) / p̃^r(θ_i)`.
- **SNPSE-B**: reweight each training sample in the DSM loss by
  `w(θ_0) = p(θ_0) / p̃^r(θ_0)`.
- **SNPSE-C**: decomposed proposal posterior score
  `s̃_ψ^r = s_ψ + ∇_θ log p̃_t^r(θ_t) - ∇_θ log p_t(θ_t)`. Implemented for
  completeness; the paper notes it frequently underperformed.

### NLSE

Learns the likelihood score with the decomposition:

```
∇_θ log p_t(θ_t|x) = ∇_θ log p_t(x|θ_t) + ∇_θ log p_t(θ_t)
```

The likelihood-score network minimizes:

```
J_lik_DSM = 1/2 ∫ λ_t E[ ||s_ψ_lik + ∇_θ log p_t(θ_t) - ∇_θ log p_{t|0}(θ_t|θ_0)||² ] dt
```

and the posterior score is `s_ψ_post = s_ψ_lik + ∇_θ log p_t(θ_t)`.

---

## Running experiments

All commands are run from the repository root (the directory containing
`npse/`). Use `python -m` module invocations.

### Quick smoke test / unit validation

```bash
pytest npse/tests/
```

### Non-sequential NPSE benchmarks (Figure 2 / Appendix F)

```bash
python -m npse.experiments.train_npse_benchmarks
```

Options:

```bash
python -m npse.experiments.train_npse_benchmarks \
  --benchmarks gaussian_linear slcp \
  --budgets 1000 10000 \
  --sde-type ve \
  --seed 0
```

Results are written to `npse/results/npse_benchmarks/npse_results.json`.

### Sequential TSNPSE benchmarks (Figure 3)

```bash
python -m npse.experiments.train_tsnpse_benchmarks \
  --benchmarks slcp lotka_volterra \
  --budgets 10000 \
  --rounds 10
```

Results are written to `npse/results/tsnpse_benchmarks/tsnpse_results.json`.

### SNPSE variant ablation (Figure 6)

```bash
python -m npse.experiments.run_snpse_variants \
  --benchmarks slcp gaussian_linear_uniform \
  --variants a b c \
  --total-budget 10000 \
  --rounds 10
```

### NPSE vs NLSE comparison (Figure 5)

```bash
python -m npse.experiments.run_nlse_comparison \
  --benchmarks gaussian_linear gaussian_linear_uniform gaussian_mixture two_moons \
  --budgets 10000
```

### Pyloric-network experiment (Figure 4 / Figure 7)

```bash
python -m npse.experiments.run_pyloric \
  --rounds 9 \
  --initial-budget 30000 \
  --per-round-budget 20000
```

If observed data is available at `npse/data/pyloric_observed.npy`, it is loaded
automatically; otherwise the script uses prior-predictive observation data with
a deterministic fallback simulator.

### Aggregate results and figures

```bash
python -m npse.experiments.evaluate_benchmarks
python -m npse.experiments.make_figures --figures 2 3 5 6 4
```

---

## Benchmark tasks

| Benchmark                | `θ` dim | `x` dim | Prior                          |
|--------------------------|---------|---------|--------------------------------|
| Gaussian Linear          | 10      | 10      | `N(0, 0.1 I)`                  |
| Gaussian Mixture         | 2       | 2       | `U(-10, 10)²`                  |
| Two Moons                | 2       | 2       | `U(-1, 1)²`                    |
| Gaussian Linear Uniform  | 10      | 10      | `U(-1, 1)¹⁰`                   |
| Bernoulli GLM            | 10      | 10      | Gaussian + precision penalty   |
| SLCP                     | 5       | 8       | `U(-3, 3)⁵`                    |
| SIR                      | 2       | 10      | LogNormal                      |
| Lotka-Volterra           | 4       | 20      | LogNormal                      |

### Gaussian Linear

```math
p(θ) = N(0, 0.1 I),   p(x|θ) = N(θ, 0.1 I)
```

The posterior is analytic: `N(x/2, 0.05 I)`. This task is used for unit
validation of sampling and density evaluation.

### Gaussian Mixture

```math
p(θ) = U(-10, 10)²,   p(x|θ) = 0.5 N(x|θ, I) + 0.5 N(x|θ, 0.01 I)
```

### Two Moons

```math
x = [r cos α + 0.25; r sin α] + [-|θ₁+θ₂|/√2; (-θ₁+θ₂)/√2]
```
with `α ~ U(-π/2, π/2)` and `r ~ N(0.1, 0.01²)`.

### Gaussian Linear Uniform

```math
p(θ) = U(-1, 1)¹⁰,   p(x|θ) = N(θ, 0.1 I)
```

### Bernoulli GLM

`θ = (β, f) ∈ R¹⁰` with `β ~ N(0, 2)` and `f ~ N(0, (FᵀF)⁻¹)` where `F`
penalizes second-order differences. The observations are 10 Bernoulli
sufficient statistics.

### SLCP

`θ ∈ R⁵` with `p(θ) = U(-3, 3)⁵`. The 8-dimensional observation consists of
four bivariate Gaussian draws whose mean and covariance are nonlinear functions
of `θ`, producing four symmetric posterior modes. Prefers the `sbibm`
implementation when available.

### SIR

`θ = (β, γ)` with `β ~ LogNormal(log 0.4, 0.5)` and
`γ ~ LogNormal(log 0.8, 0.2)`. Simulates SIR ODEs and observes 10 noisy
Binomial recordings of `I/N`.

### Lotka-Volterra

`θ = (α, β, γ, δ)` with `α, γ ~ LogNormal(-0.125, 0.5)` and
`β, δ ~ LogNormal(-3, 0.5)`. Observes 10 recordings of predator and prey
populations from ODE dynamics.

---

## Evaluation metrics

- **C2ST** — Classifier 2-Sample Test. Trains an MLP classifier to distinguish
  approximate posterior samples from reference posterior samples. Accuracy near
  `0.5` indicates a good fit.
- **MMD** — Maximum Mean Discrepancy with RBF kernel (median heuristic).
- **Simulation-Based Calibration Coverage (SBCC)** and posterior predictive
  checks for the pyloric experiment.

---

## Expected results

- C2ST on Gaussian Linear should be near `0.5`.
- For all benchmarks, C2ST should decrease as the simulation budget increases
  from 1 000 → 10 000 → 100 000.
- TSNPSE should achieve lower C2ST than non-sequential NPSE on hard benchmarks
  (notably SLCP and Lotka-Volterra).
- NPSE is expected to be particularly strong on SLCP and Lotka-Volterra,
  providing accurate and robust alternatives to NPE.
- SNPSE variant ablation (Figure 6) expects TSNPSE to outperform SNPSE-A and
  SNPSE-B on SLCP and Gaussian Linear Uniform.
- The pyloric run should reach roughly 80% valid summary statistics and produce
  posterior predictive samples close to the observed data.

---

## Notes on missing details

Where the paper is ambiguous in the accessed text, this implementation uses:

- Score-matching weighting `λ_t = g(t)²`.
- RK45 default tolerances from `torchdiffeq` (`rtol=1e-5`, `atol=1e-5`).
- `sbibm` reference simulators and posteriors for complex tasks when available,
  with self-contained fallbacks.
- The sample-based proposal-prior approximation described in Appendix C.2.3 for
  SNPSE-A proposal-prior density when more than two rounds are used.
