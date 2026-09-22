# Batch and Match (BaM) — Reproduction

Reproduction of

> **Batch and Match: Black-Box Variational Inference with a Score-Based Divergence**
> (BaM: a score-based / weighted-Fisher divergence for BBVI with full-covariance
> Gaussian variational families, minimized in closed form by a proximal *match* step.)

This repository implements the paper's **Algorithm 1 (BaM)** together with every
baseline (**ADVI**, **Score-ADVI**, **Fisher-ADVI**, **GSM**), all four experimental
settings (Gaussian targets, sinh-arcsinh targets, PosteriorDB models, CIFAR-10 VAE)
and the metrics needed to reproduce Figures 5.1–5.4 and Appendix E.3/E.4/E.6.

---

## 1. Method in one page

The variational family is a **full-covariance Gaussian** `q = N(μ, Σ)`. BaM minimizes
the **score-based divergence**

```
D(q; p) = E_q[ (∇log q(z) − ∇log p(z))^T Σ (∇log q(z) − ∇log p(z)) ]        (eq. 2)
```

which satisfies `D(q; p) = 0 ⇔ q = p`. At each iteration BaM takes a **batch step**
(draw `z_b ~ q_t`, evaluate the target score `g_b = ∇ log p(z_b)`, form the empirical
statistics `z̄, C, ḡ, Γ`) and a **match step**, which solves the *regularized and
proximal* problem exactly.

With `λ_t ≥ 0` the inverse regularization strength, define

```
U = λ_t Γ + (λ_t / (1 + λ_t)) ḡ ḡᵀ                                    (eq. 11)
V = Σ_t + λ_t C + (λ_t / (1 + λ_t)) (μ_t − z̄)(μ_t − z̄)ᵀ               (eq. 12)
```

The covariance update is the solution `X = Σ_{t+1}` of the quadratic matrix equation

```
X U X + X = V        ⇒      Σ_{t+1} = 2 V [ I + (I + 4 U V)^{1/2} ]^{-1}   (eq. 8, Lemma B.1)
```

and **only then** the mean is updated using the *new* covariance:

```
μ_{t+1} = (1 / (1 + λ_t)) μ_t + (λ_t / (1 + λ_t)) (Σ_{t+1} ḡ + z̄)           (eq. 13)
```

> **Order matters.** The mean update depends on `Σ_{t+1}`, not `Σ_t`.
> The covariance is re-symmetrized / re-projected onto the SPD cone after every update.

**Learning-rate schedules** (`λ` is an *inverse* regularization, so large `λ` = weak
regularization = fast movement):

| setting                     | schedule                 |
|-----------------------------|--------------------------|
| Gaussian targets (5.1/E.3)   | `λ_t = B·D`              |
| sinh-arcsinh (5.1/E.4)       | `λ_t = B·D/(t+1)`        |
| PosteriorDB (5.2/E.6)        | `λ_t = B·D/(t+1)`        |
| VAE (5.3)                    | constant, per-batch-size |

Ablated schedules `λ_t = B·D/√(t+1)` and `λ_t = B/(t+1)` are also exposed.

The low-rank variant (used when `B ≪ D`, `U = Q Qᵀ`) is **Lemma B.3**:

```
Σ_{t+1} = V − V Q [ (1/2) I + (Qᵀ V Q + (1/4) I)^{1/2} ]^{-2} Qᵀ V
```

costing `O(K D² + K³)` instead of `O(D³)`.

---

## 2. Repository layout

```
bam_repro/
├── bam/
│   ├── matrix_equations.py    # X U X + X = V : dense + low-rank solvers, SPD helpers
│   ├── bam.py                 # Algorithm 1: batch step, match step, mean/cov updates
│   ├── vi_base.py             # shared N(μ, Σ) state, reparameterized sampling, Gaussian KL
│   └── learning_rate.py       # λ schedules: BD, BD/(t+1), BD/√(t+1), B/(t+1)
├── targets/
│   ├── gaussian_target.py     # N(μ*, Σ*=AAᵀ): analytic score + closed-form KL
│   ├── sinh_arcsinh.py        # sinh-arcsinh normal (skew s, tail τ), score via chain rule
│   ├── posteriordb_target.py  # BridgeStan-backed score (ark, gp-pois-regr, eight-schools)
│   └── vae_target.py          # CIFAR-10 decoder Ω(·,θ̂); score of p(z|x), σ² = 0.1
├── baselines/
│   ├── advi.py                # Algorithm 2 (negative ELBO + Adam), GradientVI engine
│   ├── score_advi.py          # ADVI with the score-based divergence loss
│   ├── fisher_advi.py         # ADVI with the (unweighted) Fisher divergence loss
│   └── gsm.py                 # Algorithm 3 (ρ-solve, per-sample updates)
├── metrics/
│   ├── kl_metrics.py          # forward KL(p;q), reverse KL(q;p), KL curves
│   └── posteriordb_metrics.py # relative mean/SD error, reconstruction MSE
├── experiments/
│   ├── exp_gaussian.py        # §5.1 / Fig 5.1 + E.3  (D = 4, 16, 64, 256)
│   ├── exp_non_gaussian.py    # §5.1 / Fig 5.2 + E.4  (sinh-arcsinh, D = 10)
│   ├── exp_posteriordb.py     # §5.2 / Fig 5.3 + E.6  (B = 8, 32)
│   └── exp_vae.py             # §5.3 / Fig 5.4        (CIFAR-10, B = 10, 100, 300)
├── configs/                   # gaussian.yaml, non_gaussian.yaml, posteriordb.yaml, vae.yaml
├── scripts/run_all.py         # single entry point dispatching all experiments
├── tests/
│   ├── test_matrix_equations.py   # X U X + X = V, SPD, dense-vs-low-rank
│   └── test_bam_gsm_limit.py      # λ→∞ limit = GSM, λ→0 fixed point, 1-step recovery
├── requirements.txt
└── README.md
```

---

## 3. Installation

Python 3.10+.

```bash
pip install -r bam_repro/requirements.txt
```

Core dependencies: `numpy`, `scipy`, `jax`/`jaxlib` (CPU wheels suffice for the
synthetic and PosteriorDB experiments), `matplotlib`, `pyyaml`.
Optional: `flax`/`optax` (VAE extras — note that `targets/vae_target.py` also ships a
self-contained NumPy implementation and needs neither), `bridgestan` (+ a Stan
toolchain) for real PosteriorDB targets, `pytest` for the test suite.

A CUDA GPU is only *recommended* for the CIFAR-10 VAE and the largest-batch BaM runs.
The CPU implementation is complete; **wallclock time is out of scope** — every figure
uses *number of gradient evaluations* as the cost axis.

---

## 4. Quick verification

The load-bearing numerical primitive is validated first:

```bash
python -m pytest bam_repro/tests/test_matrix_equations.py -v
python -m pytest bam_repro/tests/test_bam_gsm_limit.py -v
```

`test_matrix_equations.py` checks that the returned `X` satisfies `X U X + X ≈ V`,
is symmetric and positive definite, and that the low-rank solver agrees with the dense
solver for random low-rank `U`.

`test_bam_gsm_limit.py` checks the analytical relationships:

* **GSM limit**: with `B = 1` and `λ_t → ∞`, BaM reproduces GSM exactly
  (`Σ_{t+1} g gᵀ Σ_{t+1} + Σ_{t+1} = Σ_t + (μ_t − z)(μ_t − z)ᵀ`,
  `μ_{t+1} = Σ_{t+1} g + z`).
* **λ → 0**: `Σ_{t+1} = Σ_t`, `μ_{t+1} = μ_t` (no movement).
* **One-step recovery** (Corollary D.5): with a large batch *and* large `λ_0`, BaM
  recovers `μ*, Σ*` of a Gaussian target in a single iteration.
* Infinite-batch Gaussian statistics (Lemma D.2) and the `ρ` positive root of GSM.

A very cheap smoke test of the whole pipeline:

```bash
python -m bam_repro.scripts.run_all --quick
```

---

## 5. Reproducing the paper's experiments

Everything is dispatched from one entry point (config-driven):

```bash
python bam_repro/scripts/run_all.py                      # all four experiments
python bam_repro/scripts/run_all.py -e gaussian           # one experiment
python bam_repro/scripts/run_all.py -e gaussian -c configs/gaussian.yaml
python bam_repro/scripts/run_all.py -o results            # writes run_all_summary.json
```

Individual experiments can also be run directly:

```bash
python -m bam_repro.experiments.exp_gaussian     --dims 4 16 64 256 --runs 10
python -m bam_repro.experiments.exp_non_gaussian --runs 10
python -m bam_repro.experiments.exp_posteriordb  --runs 5
python -m bam_repro.experiments.exp_vae          --runs 5
```

Results (`*.json`) and figures (`fig_*.png`) are written to `results/`.

### 5.1 Gaussian targets — Figure 5.1 (forward KL) and E.3 (reverse KL)

* **Dimension** `D ∈ {4, 16, 64, 256}`; target `p = N(μ*, Σ*)` with `Σ* = A Aᵀ`
  for a random `D×D` matrix `A`; analytic score `s(z) = Σ*⁻¹(μ* − z)`.
* **Initialization** `μ_0 ~ Uniform[0, 0.1]^D`, `Σ_0 = I`; **10 runs** per cell.
* **BaM**: `λ_t = B·D`, batch sizes `B = {20, 40}`.
* **Baselines**: batch size `B = 2` (except BaM), grid-searched Adam learning rates —
  ADVI `0.01`, Fisher `0.01`, Score `{0.01, 0.005, 0.001, 0.001}` for `D = {4, 16, 64, 256}`.
* **Metrics**: forward `KL(p;q)` and reverse `KL(q;p)` in closed form, vs. gradient evaluations.

Expected behaviour: BaM converges in **orders of magnitude fewer gradient evaluations**
than ADVI; GSM is competitive in some cells; BaM improves with larger batch sizes while
GSM saturates beyond `B = 2`.

### 5.2 sinh-arcsinh targets — Figure 5.2 (forward KL) and E.4 (reverse KL)

* Non-Gaussian target at `D = 10`:
  `z = sinh( (1/τ)( sinh⁻¹(y) + s ) )` with `y ~ N(μ, Σ)`; `s` = skew, `τ` = tail weight.
* Six settings: `τ = 1` with `s ∈ {0.2, 1.0, 1.8}` and `s = 0` with `τ ∈ {0.1, 0.9, 1.7}`.
* **BaM**: `λ_t = B·D/(t+1)`, batch sizes `B ∈ {2, 5, 10, 20, 40}`.
* **Baselines**: `B = 5`; ADVI lr `0.02`, Fisher lr `0.05`, Score per-setting.
* **10 runs**; forward/reverse KL via Monte-Carlo using the known log densities.

Expected behaviour: BaM faster than ADVI; for large skew (`s = 1.0, 1.8`) BaM reaches a
higher forward KL but a similar reverse KL; GSM/Score diverge at `s = 1.8`
(flagged as `diverged` in the result records).

### 5.3 PosteriorDB models — Figure 5.3 and E.6

* Models: **ark** (`D = 7`, Gaussian), **gp-pois-regr** (`D = 13`),
  **eight-schools-centered** (`D = 10`); target
  `p(z | {x_n}) ∝ p(z) p({x_n} | z)` with the score provided by BridgeStan.
* Batch sizes `B ∈ {8, 32}`; **5 runs**; BaM `λ_t = B·D/(t+1)`; ADVI lr `0.02`,
  Fisher lr `0.05`, Score per-model.
* **Metrics**: relative mean error `‖(μ − μ̂)/σ‖₂` and relative SD error
  `‖(σ − σ̂)/σ‖₂` against HMC reference moments.

Expected behaviour: BaM has lower relative mean error than ADVI at both `B = 8, 32`;
GSM converges faster at small `B` but oscillates; BaM converges to a larger relative SD
error on *eight-schools*.

> **BridgeStan note.** `targets/posteriordb_target.py` resolves the target through a
> three-tier fallback: real BridgeStan model → cached Stan model + JSON HMC draws →
> a flagged `SurrogateGaussianTarget`. The surrogate path lets the metric/plotting
> pipeline run on a machine without a Stan toolchain; results produced that way are
> marked `is_surrogate=True` in the JSON and must not be reported as paper numbers.
> Put Stan models / HMC draws in the cache directory (env var `BAM_POSTERIORDB_CACHE`)
> to get real targets.

### 5.4 CIFAR-10 VAE — Figure 5.4

* Deep generative model `z_n ~ N(0, I)`, `x_n | z_n ~ N(Ω(z_n, θ̂), σ² I)` with
  `σ² = 0.1`, `z ∈ R^256`, `x ∈ R^3072` (CIFAR-10, images scaled to `[-1, 1]`).
* The decoder `Ω(·, θ̂)` is pre-trained by variational EM (factorized-Gaussian encoder,
  Adam with `lr: 0 → 1e-4` over 100 warmup steps, then `1e-4 → 1e-5` over 500 decay
  steps, 100 epochs, `mc_sim = 1`). The trained decoder is cached as
  `cache/vae_decoder.pkl` so the EM cost is paid once.
* Target score for a held-out image `x'`:
  `∇_z [ log p(z') + log p(x' | z') ] = −z' + (1/σ²) J_Ω(z')ᵀ (x' − Ω(z'))`.
* Batch sizes `B ∈ {10, 100, 300}`; a short pilot (`T = 100`) selects the learning rate
  (baselines) or `λ` (BaM) before the full run (`T = 1000`); ADVI lr `0.02`.
* **Metric**: reconstruction MSE vs. gradient evaluations. The amortized-VI (AVI)
  reference reconstruction is also reported (`report_avi: true`).

Expected behaviour: BaM is poor at `B = 10`, competitive at moderate `B`, and at
`B = 300` converges **≥ 1 order of magnitude faster** than ADVI/GSM; both BaM and ADVI
beat the AVI reconstruction.

> **CIFAR-10 note.** `load_cifar10` looks for the standard `cifar-10-batches-py`
> directory; if absent it falls back to a documented synthetic-image surrogate
> (`allow_synthetic: true`) so the pipeline remains runnable. Download CIFAR-10 and
> place it under `data/` (or pass `data_root=`) for real numbers.

---

## 6. Protocol and reporting conventions

* Curves report the **mean over runs** with standard errors:
  10 runs for the synthetic experiments (§5.1), 5 runs for PosteriorDB (§5.2) and the
  VAE (§5.3).
* The x-axis is always the **number of gradient evaluations**
  (`grad_evals = B × iterations`), never wallclock — wallclock depends on
  implementation/JIT/hardware and is explicitly out of scope.
* Learning rates for the gradient-based baselines are **grid searched** (in scope),
  via the `learning_rate_grid` entries in `configs/*.yaml`.
* BaM is essentially hyperparameter-free: only `λ_t` (the schedule) is chosen, and the
  paper's schedule is fixed per experiment family.
* Reported diverged runs (non-finite or enormous KL) are kept in the JSON with
  `diverged: true` rather than being silently dropped.

### Out of scope

The following are intentionally **not** reproduced because they depend on wallclock
timing or are ablated variants of already-covered experiments: the timing panels of
E.1/E.7, and the BaM learning-rate schedule sweeps (E.2, E.5).

---

## 7. Documented defaults for unspecified details

Where the paper is silent, the following defaults are used (each is also recorded in
the corresponding config file):

| detail | default chosen |
|---|---|
| Gaussian target mean | `μ* ~ Uniform[0, 0.1]^D`; `A ~ N(0, I)`, `Σ* = A Aᵀ` (optional condition-number rescale) |
| iteration counts `T` | run to a per-setting **gradient-evaluation budget** (20k–400k as configured) so the metric plateaus |
| Fig 5.1/5.2 BaM batch sizes | Gaussian `B = {20, 40}`; non-Gaussian `B = {2, 5, 10, 20, 40}` |
| GSM `ρ` solve | stable positive root `ρ = (√(1 + 4c) − 1)/2` with `c = sᵀΣs + [(μ − z)ᵀs]²`, guarded by `max(c, 0)` |
| VAE hidden width | `c_hid = 64` |
| VAE pre-training batch size | `128` |
| CIFAR-10 scaling | images scaled to `[-1, 1]` to match the `tanh` decoder output |
| Monte-Carlo KL samples | `4096` |

---

## 8. Interfaces at a glance

```python
from bam_repro.bam import BaM, make_schedule
from bam_repro.targets import random_gaussian_target

target = random_gaussian_target(dim=16, rng=0)          # p = N(mu*, Sigma*)
bam = BaM(mu0=..., Sigma0=..., score_fn=target.score,
          batch_size=20, lam=make_schedule("BD", batch_size=20, dim=16))
result = bam.run(T=500)                                  # result.mu, result.Sigma
```

Every target exposes the black-box contract used by BaM and the baselines:

```python
target.score(z)      # ∇_z log p(z)          (D,) or (B, D)
target.log_prob(z)   # log p(z)
target.sample(n, rng=...)   # draws from the target (for MC metrics)
```

Every learner exposes `run(T)`, `grad_evals`, `mu_history`/`Sigma_history`, so the
experiment drivers can plot any metric against gradient evaluations uniformly.
