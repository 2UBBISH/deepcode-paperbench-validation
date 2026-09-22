# Batch and Match: Score-Based Black-Box Variational Inference

Reproduction of the paper **"Batch and Match: Score-Based Black-Box Variational Inference"**.

This repository implements the Batch-and-Match (BaM) algorithm for full-covariance
Gaussian variational inference, along with the competing baselines evaluated in the
paper: ADVI, Score, Fisher, and GSM. It also provides the synthetic target
distributions, hierarchical posterior wrappers, and deep generative model experiments
used in Sections 5.1-5.3.

---

## Overview

Batch-and-Match is a score-based variational inference method. Given a Gaussian
variational family `q(z) = N(mu, Sigma)`, BaM minimizes the score-based divergence

```
D(q; p) = E_q[ (grad log q(z) - grad log p(z))^T Gamma_q^{-1}
               (grad log q(z) - grad log p(z)) ],
```

where `Gamma_q = E_q[(grad log q)(grad log q)^T]`. For a Gaussian `q` this reduces to

```
D(q; p) = E_q[ || grad log q(z) - grad log p(z) ||_Sigma^2 ].
```

Each iteration of BaM samples a batch from the current Gaussian, computes target
scores, and applies a closed-form proximal MATCH update. The covariance update solves
the quadratic matrix equation

```
Sigma U Sigma + Sigma = V
```

in closed form (full-rank `O(D^3)` or low-rank `O(K D^2 + K^3)`).

---

## Repository Layout

```
project_root/
├── README.md
├── requirements.txt
├── src/bam/
│   ├── __init__.py
│   ├── quadratic_solver.py      # Core full/low-rank quadratic matrix solver
│   ├── batch_stats.py           # Batch sample/score statistics
│   ├── divergences.py           # Score-based divergence, KL, Fisher diagnostics
│   ├── match_update.py          # Closed-form mean/covariance MATCH updates
│   ├── algorithm.py             # BaM Batch-and-Match Algorithm 1
│   ├── baselines.py             # ADVI, Score, Fisher, GSM algorithms
│   ├── targets.py               # Gaussian and sinh-arcsinh targets
│   ├── posterior_targets.py     # posteriordb/BridgeStan posterior wrappers
│   ├── deep_generative.py       # CIFAR-10 VAE decoder/encoder and AVI helpers
│   └── utils.py                 # PRNG, timing, metrics, plotting helpers
├── experiments/
│   ├── run_gaussian_targets.py  # Section 5.1 Gaussian targets
│   ├── run_sinh_arcsinh.py      # Section 5.1 non-Gaussian skew/tail targets
│   ├── run_posteriordb.py       # Section 5.2 hierarchical Bayesian models
│   ├── run_cifar.py             # Section 5.3 deep generative model
│   └── plot_results.py          # Figure reproduction
└── tests/
    ├── test_quadratic_solver.py
    ├── test_batch_stats.py
    ├── test_divergences.py
    ├── test_match_update.py
    ├── test_gaussian_convergence.py
    └── test_baselines.py
```

---

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The hierarchical posterior experiments (Section 5.2) additionally require
`posteriordb`, `bridgestan`, and `stanio`, which are listed in `requirements.txt`.
These are optional at import time: the module `src/bam/posterior_targets.py` raises an
informative error only when a posterior target is actually loaded without the
dependencies installed.

For the CIFAR-10 experiments, a GPU-compatible JAX installation is recommended but not
required. CPU is sufficient for the low-dimensional examples.

---

## Running the Experiments

### 1. Unit tests

```bash
pytest tests/
```

The test suite validates:

- The quadratic solver (`X U X + X = V`), including full-rank/low-rank agreement,
  symmetry, positive semidefiniteness, and zero-`U` limiting behavior.
- Batch statistics against known Gaussian moments.
- Score-based divergence closed forms, empirical estimates, affine invariance, and
  Gaussian KL/Fisher formulas.
- BaM MATCH updates, the BaM proximal objective, and the GSM limiting case.
- Gaussian-target convergence behavior described by Theorem 1.
- ADVI, Score, Fisher, and GSM baseline step updates.

### 2. Section 5.1 - Gaussian targets

```bash
python experiments/run_gaussian_targets.py \
  --dims 4,16,64,128,256 \
  --algos bam,advi,score,fisher,gsm \
  --T 500 \
  --B 32 \
  --runs 5 \
  --seed 0 \
  --threshold 0.01 \
  --out results/gaussian_targets.csv
```

BaM uses the constant learning rate `lambda_t = B * D` for Gaussian targets. The
script records forward/reverse KL histories, normalized mean/covariance errors,
gradient evaluations to reach a KL threshold, and wall-clock time.

### 3. Section 5.1 - Sinh-arcsinh (skew/tail) targets

```bash
python experiments/run_sinh_arcsinh.py \
  --dims 4 \
  --T 200 \
  --B 32 \
  --runs 5 \
  --seed 0 \
  --out results/sinh_arcsinh.csv
```

The script evaluates skew cases (`tau=1`, `s in {0.2, 1.0, 1.8}`) and tail cases
(`s=0`, `tau in {0.1, 0.9, 1.7}`). BaM uses the decaying learning rate
`lambda_t = B * D / (t + 1)`. Forward and reverse KL are estimated by Monte Carlo.

### 4. Section 5.2 - Hierarchical posteriors

```bash
# Real posteriordb/BridgeStan models (requires posteriordb + bridgestan)
python experiments/run_posteriordb.py \
  --models ark,gp-pois-regr,eight-schools-centered \
  --algos bam,advi,gsm \
  --batch-sizes 8,32 \
  --T 500 \
  --runs 5 \
  --seed 0 \
  --out results/posteriordb.csv

# Smoke test with synthetic Gaussian stand-ins (no external data needed)
python experiments/run_posteriordb.py --synthetic --out results/posteriordb_synthetic.csv
```

The script reports relative posterior mean and SD errors against HMC reference draws.
BaM uses the decaying learning rate `lambda_t = B * D / (t + 1)`.

### 5. Section 5.3 - CIFAR-10 deep generative model

```bash
python experiments/run_cifar.py \
  --test-size 5 \
  --batch-sizes 10,300 \
  --advi-batch-size 10 \
  --gsm-batch-size 10 \
  --T-pilot 100 \
  --T-final 1000 \
  --epochs 10 \
  --seed 0 \
  --out results/cifar.csv
```

The script trains/loads a convolutional VAE with a `256`-dimensional latent space and
fixed decoder variance `sigma^2 = 0.1`, then runs full-covariance posterior inference
with BaM, ADVI, and GSM plus a factorized amortized AVI baseline. It performs a pilot
run to select learning rates, then final runs of `1000` iterations, and reports
reconstruction MSE and wall-clock time. A fixed-gradient-budget comparison
(`3000` gradient evaluations; ADVI `B=10, T=300` vs BaM `B=300, T=10`) is included.

### 6. Figure reproduction

```bash
python experiments/plot_results.py --all
# or select specific figure families
python experiments/plot_results.py --figures gaussian sinh posteriordb cifar wallclock
```

The plotting script consumes the CSV outputs in `results/` and writes PNG figures to
`figures/`.

---

## Key Implementation Details

- **Quadratic solver** (`src/bam/quadratic_solver.py`):
  - Full-rank: `X = 2 V [ I + (I + 4 U V)^{1/2} ]^{-1}`.
  - Low-rank (`U = Q Q^T`):
    `X = V - V Q [ 0.5 I + (Q^T V Q + 0.25 I)^{1/2} ]^{-2} Q^T V`.
  - A dispatcher selects the low-rank solver when the numerical rank of `U` is at
    most `D // 2`.

- **BaM MATCH update** (`src/bam/match_update.py`):
  ```
  U = lam * Gamma + lam / (1 + lam) * gbar gbar^T
  V = Sigma_t + lam * C + lam / (1 + lam) * (mu_t - zbar)(mu_t - zbar)^T
  Sigma_{t+1} solves Sigma U Sigma + Sigma = V
  mu_{t+1} = lam / (1 + lam) * (zbar + Sigma_{t+1} gbar) + 1 / (1 + lam) * mu_t
  ```

- **Learning-rate schedules**:
  - Gaussian targets: `lambda_t = B * D` (constant).
  - Non-Gaussian and posterior targets: `lambda_t = B * D / (t + 1)` (decaying).
  - GSM is recovered as the `B = 1`, `lambda -> infinity` limit of BaM.

---

## References

- Batch and Match: Score-Based Black-Box Variational Inference. The reference paper
  defining the score-based divergence, the BaM proximal MATCH update, and the
  Gaussian-target convergence analysis.
- posteriordb: a posterior database for reproducible Bayesian computation.
- BridgeStan: efficient in-memory access to Stan model log densities and gradients.
- CIFAR-10: Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny
  Images.

---

## Notes on Reproduction

Where exact random seeds are absent in the paper, this implementation uses fixed seeds
and reports means/standard errors over multiple runs. The neural architecture for the
CIFAR-10 VAE follows the plan's default: a convolutional encoder/decoder with a
dense latent layer of dimension `256` and constant decoder variance `sigma^2 = 0.1`.
For posteriordb experiments, provided HMC reference draws are used to compute relative
mean/SD errors against BridgeStan-exposed target densities.
