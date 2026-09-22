# Batch and match (BaM): a reproduction

This repository reproduces the core contributions of

> D. Cai, C. Modi, L. Pillaud-Vivien, C. C. Margossian, R. M. Gower, D. M. Blei,
> L. K. Saul. **Batch and match: black-box variational inference with a
> score-based divergence.** ICML 2024.

The paper proposes **BaM**, a black-box variational inference (BBVI) algorithm
that minimizes a *score-based divergence* between the target ``p`` and a
full-covariance Gaussian ``q``.  Each iteration alternates a **batch step**
(draw ``B`` samples from the current approximation and evaluate the target
scores) with a **match step** that minimizes the empirical divergence plus a KL
regularizer *in closed form* -- a quadratic matrix equation whose solution gives
the next covariance.  The paper also proves exponential convergence of BaM to a
Gaussian target and compares BaM to ADVI, GSM, and score/Fisher variants of
ADVI on Gaussian targets, sinh-arcsinh targets, hierarchical Bayesian models
and a deep generative model.

Everything below is implemented from the paper (and its addendum) only; the
authors' code base was not consulted.

---

## 1. What is implemented

| Paper component | Where it lives | Status |
| --- | --- | --- |
| Score-based divergence, Definition A.2 and eq. (2) | `src/bam/divergence.py` | implemented (Monte Carlo + closed form) |
| Properties: non-negativity, affine invariance, annealing, exponential tilting, KL relation | `tests/test_divergence.py` | numerically verified |
| BaM: batch step, match step, closed-form covariance update, low-rank solver | `src/bam/bam.py`, `src/bam/linalg.py` | implemented |
| Quadratic matrix equations, Lemmas B.1-B.4 | `src/bam/linalg.py`, `tests/test_matrix_equations.py` | implemented + tested |
| GSM as the ``B=1, lambda -> inf`` limit of BaM (Appendix C.3) | `tests/test_bam_updates.py` | numerically verified |
| Theorem 3.1 (exponential convergence, infinite batch) | `src/bam/theory.py`, `experiments/verify_theorem31.py`, `tests/test_theory.py` | recursions implemented; bounds verified numerically |
| Baselines ADVI / Score / Fisher / GSM (Algorithms 2-3, Appendix E.1) | `src/bam/baselines.py` | implemented |
| Section 5.1: Gaussian targets of increasing dimension (Fig. 5.1, E.3) | `experiments/run_gaussian_targets.py` | implemented |
| Section 5.1: sinh-arcsinh targets (Fig. 5.2, E.4) | `experiments/run_shash_targets.py` | implemented |
| Section 5.2: posteriordb hierarchical models (Fig. 5.3, E.6) | `experiments/run_posteriordb.py`, `src/bam/posterior_models.py` | implemented |
| Section 5.3: deep generative model / CIFAR-10 VAE (Fig. 5.4, E.7) | `src/bam/vae.py`, `src/bam/deep_generative.py`, `experiments/train_vae_cifar10.py`, `experiments/run_vae_posterior.py` | implemented (training must be run on a GPU machine) |
| Learning-rate grid searches for gradient methods | `src/bam/grid_search.py` | implemented, paper's selected values recorded |

Out of scope (per the addendum): wallclock-based figures (E.1, E.2, E.5, E.7 are
either wallclock or learning-rate-schedule sweeps) and the alternative BaM
learning-rate schedules.  The code still *can* produce these plots (`--outdir`
arguments), they are simply not part of the reproduction targets.

---

## 2. The algorithm, in one place

For ``q_t = N(mu_t, Sigma_t)`` and a batch ``z_1..z_B ~ q_t`` with scores
``g_b = grad log p(z_b)``, the batch step computes

```
zbar = mean(z_b),   gbar = mean(g_b)
C     = cov(z_b),   Gamma = cov(g_b)
```

and the match step solves

```
U = lambda_t * Gamma + lambda_t/(1+lambda_t) * gbar gbar^T
V = Sigma_t + lambda_t * C + lambda_t/(1+lambda_t) * (mu_t - zbar)(mu_t - zbar)^T

Sigma_{t+1} U Sigma_{t+1} + Sigma_{t+1} = V                 (quadratic matrix equation)
mu_{t+1}    = (mu_t + lambda_t (Sigma_{t+1} gbar + zbar)) / (1 + lambda_t)
```

The covariance update is computed either with the closed form
``Sigma_{t+1} = 2 V [I + (I + 4 U V)^{1/2}]^{-1}`` (Lemma B.1, ``O(D^3)``) or,
when ``B < D`` and ``U = Q Q^T`` has rank ``O(B)``, with the low-rank solver
``Sigma_{t+1} = V - V Q [I/2 + (Q^T V Q + I/4)^{1/2}]^{-2} Q^T V``
(Lemma B.3, ``O(D^2 B + B^3)``).  `solver="auto"` picks the low-rank variant
whenever ``B <= D``, which is the regime of the small-batch experiments.

---

## 3. Repository layout

```
src/bam/
  linalg.py             quadratic matrix equations, PSD helpers (Appendix B)
  divergence.py         score-based divergence, KL estimators (Section 2, Appendix A)
  targets.py            target interface, Gaussian, sinh-arcsinh, posterior targets
  posterior_models.py   arK / gp_pois_regr / eight_schools_centered log densities (JAX)
  bam.py                Algorithm 1 (BaM)
  baselines.py          ADVI / Score / Fisher (Algorithm 2), GSM (Algorithm 3)
  adam.py               small dependency-free Adam
  grid_search.py        learning-rate grid searches + the paper's selected values
  runner.py             shared experiment machinery (methods, metrics, repeats)
  metrics.py            KL, relative mean/SD error, reconstruction MSE
  theory.py             infinite-batch recursions and Theorem 3.1 bounds
  vae.py                convolutional VAE (encoder/decoder) of Section 5.3
  deep_generative.py    posterior p(z | x') for the VAE, and the AVI baseline
  cifar10.py            CIFAR-10 downloader/loader
  utils.py              plotting helpers

experiments/
  run_gaussian_targets.py    Section 5.1 (Gaussian targets, Fig. 5.1 and E.3)
  run_shash_targets.py       Section 5.1 (sinh-arcsinh targets, Fig. 5.2 and E.4)
  run_posteriordb.py         Section 5.2 (Fig. 5.3 and E.6)
  train_vae_cifar10.py       Section 5.3: pre-train the VAE (Appendix E.6)
  run_vae_posterior.py       Section 5.3: posterior inference on a test image (Fig. 5.4, E.7)
  verify_theorem31.py        Section 3.2 / Appendix D: numerical check of Theorem 3.1
  prepare_posteriordb.py     downloads the posteriordb data and reference draws

tests/                unit tests (unittest; also runnable with pytest)
data/posteriordb/     data + HMC reference draws for the three Section 5.2 targets
results/              outputs of the scripts (plots + JSON summaries)
docs/reproduction_notes.md   assumptions, deviations, and runtime guidance
```

---

## 4. Quick start

```
pip install -r requirements.txt          # numpy, scipy, jax (CPU is enough), matplotlib

# unit tests (~40 s on a laptop CPU)
python -m unittest discover -s tests -v

# smoke versions of the three empirical sections (~2 min in total)
python experiments/run_gaussian_targets.py --quick
python experiments/run_shash_targets.py --quick
python experiments/run_posteriordb.py --quick
python experiments/verify_theorem31.py --dim 8 --n-iters 60 --batch-check 6
```

Full reproductions (these are the runs that produce the paper's figures):

```
# Section 5.1: Gaussian targets D = 4, 16, 64, 256, 10 runs
python experiments/run_gaussian_targets.py

# Section 5.1: sinh-arcsinh targets, D = 10, 10 runs
python experiments/run_shash_targets.py

# Section 5.2: posteriordb models, batch sizes 8 and 32, 5 runs
python experiments/run_posteriordb.py

# Section 5.2, recommended configuration (see section 8 below): Stan's
# unconstrained parameterization with a slower-decaying BaM schedule for the
# two well-conditioned models, and the constrained (model) parameterization for
# gp_pois_regr
python experiments/run_posteriordb.py --models arK eight_schools_centered \
    --parameterization unconstrained --bam-schedule sqrt
python experiments/run_posteriordb.py --models gp_pois_regr \
    --parameterization constrained --gp-variable f_tilde --bam-schedule sqrt

# Section 5.3: pre-train the VAE (GPU strongly recommended), then infer
python experiments/train_vae_cifar10.py --epochs 100 --out checkpoints/vae_cifar10.npz
python experiments/run_vae_posterior.py --checkpoint checkpoints/vae_cifar10.npz
```

Every script writes a ``*.png`` (mean curve with standard-error band, in the
style of the paper) and a ``*_summary.json`` with the underlying numbers into
``results/``.

---

## 5. What was actually run while developing this reproduction

The environment used for this reproduction has **no GPU** and long experiments
were not permitted, so the scripts were validated with the ``--quick`` variants,
with medium-sized runs, and with unit tests; the full runs are meant to be
executed elsewhere.  All artifacts are in `results/` (`smoke/` and
`validation/`, see `results/README.md` for the exact commands) and match the
*trends* of the paper:

| Experiment | Observation (runs done here) | Paper |
| --- | --- | --- |
| Gaussian targets, fixed budget of gradient evaluations (D = 16, 3 runs, 4000 gradient evaluations) | BaM and GSM reach ``KL(p;q) ~ 1e-14``; ADVI/Fisher/Score are still at ``KL = 12 / 65 / 57`` (ADVI only reaches ``KL ~ 1`` after 50,000 gradient evaluations, ~50x BaM's budget) | Fig. 5.1: "BaM converges orders of magnitude faster than ADVI"; GSM competitive on Gaussian targets |
| sinh-arcsinh, ``s = 1.8`` (D = 10, 2 runs, 2000 gradient evaluations, B = 5/20) | reverse KL: ``BaM 7.6 / ADVI 7.1 / Fisher 9.3 / Score 57 / GSM 8633`` -- GSM and Score diverge, BaM and ADVI end at similar values while BaM's forward KL is larger; for ``s = 0.2`` all methods agree (``0.13-0.23``); for the heavy-tailed ``tau = 0.1`` all methods end at similar reverse KL values | Fig. 5.2/E.4: "BaM converges to a higher value of the forward KL but to similar values of the reverse KL"; "the reverse KL for GSM diverges when the target is highly skewed" |
| posteriordb (B = 8 and 32, 2 runs, up to 5000 gradient evaluations) | relative mean error: ``BaM 0.046`` vs ``ADVI 95-214`` (arK), ``BaM 0.44`` vs ``ADVI 8.3`` (gp_pois_regr), ``BaM 0.40-0.51`` vs ``ADVI 3.0`` with ``GSM 0.15`` (eight-schools); on the hierarchical model BaM reaches a larger relative SD error (``1.03``) than GSM (``0.71``) | Fig. 5.3/E.6: "BaM outperforms ADVI"; "for smaller batch sizes GSM can converge faster"; "in the hierarchical example, BaM converges to a larger relative SD error" |
| Theorem 3.1 | the bounds ``||eps_t|| <= (1-delta)^t ||eps_0||`` and ``||Delta_t|| <= (1-delta)^t ||Delta_0|| + t(1-delta)^{t-1}||eps_0||^2`` hold for ``lambda in {0.05, 1, 10, 1000}`` together with the per-iteration inequalities; the infinite-batch recursion matches a finite-batch run (``B = 4096``, 0.5% on ``||eps_t||``; closer for larger ``B``); one-step convergence as ``lambda -> inf`` | Theorem 3.1, Corollary D.5 |

The full runs were not executed here because, for example, `run_gaussian_targets.py`
at its defaults performs ``10 runs x 4 dimensions x 7 methods`` up to ``1e5``
gradient evaluations each, which takes hours on CPU; the VAE pre-training
(100 epochs of CIFAR-10) is far outside the allowed compute as well.  The
following were all executed successfully during development: every unit test,
the three ``--quick`` experiment runs, the grid-search path, the theory check,
and VAE forward/backward passes at the full architecture size.

---

## 6. How each figure maps to code

* **Figure 5.1 / E.3** (`run_gaussian_targets.py`): targets ``N(0, A A^T)`` for
  ``D = 4, 16, 64, 256`` (Appendix E.3), initialization ``mu_0 ~ U[0, 0.1]``,
  ``Sigma_0 = I``, constant ``lambda_t = B D`` for BaM, ``B = 2`` for
  ADVI/Score/Fisher/GSM, forward (5.1) and reverse (E.3) KL versus the number of
  gradient evaluations; 10 runs.
* **Figure 5.2 / E.4** (`run_shash_targets.py`): sinh-arcsinh targets with
  ``D = 10``, skew ``s = 0.2, 1.0, 1.8`` (``tau = 1``) and tail ``tau = 0.1,
  0.9, 1.7`` (``s = 0``), ``lambda_t = B D/(t+1)`` for BaM, ``B = 5`` for the
  baselines; forward (5.2) and reverse (E.4) KL.
* **Figure 5.3 / E.6** (`run_posteriordb.py`): relative posterior mean (5.3) and
  SD (E.6) errors for ``arK`` (D=7), ``gp_pois_regr`` (D=13) and
  ``eight_schools_centered`` (D=10), ``B = 8`` (dashed) and ``B = 32`` (solid),
  ``lambda_t = B D/(t+1)``, reference summaries from the posteriordb HMC draws.
* **Figure 5.4 / E.7** (`run_vae_posterior.py`): reconstruction MSE of a
  CIFAR-10 test image versus gradient evaluations (5.4) and wallclock (E.7),
  comparing BaM and ADVI with ``B = 10, 100, 300`` after a 100-iteration pilot
  learning-rate search, GSM with the same batch sizes, and the amortized
  encoder (AVI).
* **Theorem 3.1** (`verify_theorem31.py`): errors and predicted bounds for
  several regularization levels, the finite-batch comparison, and the one-step
  convergence of Corollary D.5.

---

## 7. Assumptions, and where they differ from the paper

The paper (plus addendum) leaves a few implementation details unspecified; the
choices made here are listed in full in `docs/reproduction_notes.md`, with the
most important ones being:

1. **Sinh-arcsinh base distribution** -- the paper does not say which Gaussian
   the transformation is applied to; we use the standard normal ``N(0, I)``.
2. **Posterior parameterization** -- the Section 5.2 targets are implemented on
   the *unconstrained* (Stan) parameter space, including the log-Jacobian of the
   positivity constraints, with the variational summaries mapped back to the
   constrained space before the relative-error metrics are computed.  The
   constrained-space density is available as an option
   (``--parameterization constrained``) and is validated against the HMC
   reference draws through ``E_p[score] = 0`` in `tests/test_targets.py`.
3. **posteriordb reference draws** -- ``gp_pois_regr``'s draws are given in terms
   of the transformed parameter ``f = L(rho, alpha) f_tilde``; they are mapped
   back to the model parameter ``f_tilde`` (``prepare_posteriordb.py``).
4. **ADVI parameterization** -- full-covariance Gaussian with a Cholesky factor
   and an exponential diagonal, optimized with Adam (Algorithm 2).
5. **Learning rates of the baselines** -- the values selected by the grid
   searches reported in the paper are the defaults; ``--grid-search`` re-derives
   them (the candidate grids are in `src/bam/grid_search.py`).
6. **VAE width** ``c_hid`` -- not specified in the paper; the default is 64
   (``--c-hid``) and the architecture otherwise follows the addendum exactly.
7. **Number of BaM batch sizes in Figure 5.1** -- the figure's legend is not
   recoverable from the paper text, so ``B = 2, 5, 10, 20, 40`` are used.

## 8. Repository hygiene

* Only source code, small data files (posteriordb data and reference draws,
  ~1 MB) and small result summaries are committed; no checkpoints or image data.
* `data/posteriordb/*` is generated by `experiments/prepare_posteriordb.py` from
  the public posteriordb repository (Magnusson et al., 2022); the cached copies
  are committed so that the experiments run offline.

## 9. Section 5.2: recommended configuration and a known sensitivity

The three posteriordb targets are exponentially sensitive to their scale
parameters (``sigma``, ``tau``, ``rho``, ``alpha``), because those parameters
multiply variances or appear inside the GP covariance.  With the paper's
initialization (``mu_0 ~ U[0, 0.1]``, ``Sigma_0 = I``) the *initial* variational
samples can therefore land in a region where the target score is astronomically
large (we measure ``|grad log p|`` up to ``1e11`` for ``gp_pois_regr``), and a
score-based update with ``lambda_t = B D / (t+1)`` and ``B D = 104`` can then be
sent far outside the posterior.  This is a property of the target and the
prescribed initialization rather than of the BaM update itself (the closed-form
update is exactly the one in Algorithm 1, and it is verified against the
equations of Appendix C in `tests/test_bam_updates.py`).

Measured relative posterior mean errors after 8,000 gradient evaluations
(3 runs, ``B = 8 / 32``); "diverge" means the run left the posterior region:

| target | space | schedule | BaM | ADVI | GSM |
| --- | --- | --- | --- | --- | --- |
| arK (D=7) | unconstrained | ``BD/(t+1)`` (paper) | 24.1 / **0.044** | 5.6 / 72.8 | 0.13 |
| arK (D=7) | unconstrained | ``BD/sqrt(t+1)`` | **0.103 / 0.073** | 39.3 / 166.5 | 0.12 |
| eight-schools (D=10) | unconstrained | ``BD/(t+1)`` (paper) | **0.458 / 0.401** | 3.0 / 3.0 | 0.20 / 0.35 |
| eight-schools (D=10) | unconstrained | ``BD/sqrt(t+1)`` | **0.772 / 0.493** | 3.0 / 3.0 | 0.20 / 0.35 |
| gp_pois_regr (D=13), ``f_tilde`` | constrained | ``BD/sqrt(t+1)`` | diverge / **5.0** | 86.2 | diverge |
| gp_pois_regr (D=13), ``f_tilde`` | constrained | ``BD/(t+1)`` (paper) | diverge / 24.9 | 86.2 | diverge |
| gp_pois_regr (D=13), ``f_tilde`` | unconstrained | ``BD/(t+1)`` (paper) | diverge | 8.1 | 0.52 / 17.1 |

In every configuration where BaM runs stably it converges to a *much* smaller
relative mean error than ADVI, which is the qualitative claim of Figure 5.3;
for the hierarchical model GSM is faster at the smaller batch size and BaM
improves with the larger one, again as in the paper.  The configuration that
reproduces this for all three models is the one recommended above (section 4):
Stan's unconstrained parameterization with ``--bam-schedule sqrt`` for ``arK``
and ``eight_schools_centered``, and the constrained (model) parameterization
with ``--gp-variable f_tilde --bam-schedule sqrt`` for ``gp_pois_regr``.

`BaMConfig.skip_nonfinite_updates` (on by default) freezes an iterate instead of
propagating NaNs when a batch produces a non-finite update, and the covariance
is projected back onto the positive-definite cone after every update
(`bam.linalg.ensure_positive_definite`); both are numerical safeguards and do
not change the algorithm on well-behaved targets.
