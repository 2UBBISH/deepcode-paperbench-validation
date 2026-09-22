# Reproduction notes: assumptions, deviations, and runtime guidance

This file records every place where the paper (or its addendum) left a detail
unspecified, together with the choice made here and how it could be changed.

## 1. Sinh-arcsinh targets (Section 5.1, second part)

The paper defines the non-Gaussian targets by transforming a Gaussian random
variable,

```
z = sinh( (asinh(y) + s) / tau ),   y ~ N(mu, Sigma),
```

but does not state the base ``(mu, Sigma)``, nor how ``s`` and ``tau`` are
arranged across dimensions.  Choices made here (in
`experiments/run_shash_targets.py` and `src/bam/targets.py`):

* base ``N(0, I)`` -- the canonical sinh-arcsinh construction, which also makes
  the initial divergence of Section 5.1 comparable to the Gaussian-target case;
* the same scalar skew ``s`` and tail ``tau`` in every dimension of the ``D = 10``
  target;
* the density (including its normalizing constant) follows from the change of
  variables, and `tests/test_targets.py` checks that it integrates to one, that
  ``s = 0, tau = 1`` reduces to the Gaussian, and that the samples match the
  density.

For ``tau = 0.1`` the tails are extremely heavy (``p(z) ~ exp(-0.5 |z|^{2 tau})``
for large ``|z|``), so Monte-Carlo estimates of the KL divergences have large
variance; ``--n-eval-samples`` controls the number of samples used.

## 2. posteriordb targets (Section 5.2)

Model sources are the Stan files of posteriordb; they are transcribed into JAX
(`src/bam/posterior_models.py`) so that no Stan compilation is needed.  The
gradients are obtained by automatic differentiation, in the same way that the
paper obtains them with BridgeStan.

* **Parameterization.**  The models contain positivity constraints
  (``sigma > 0``, ``rho > 0``, ``alpha > 0``).  Stan -- and therefore BridgeStan
  and posteriordb's HMC reference draws -- works in the *unconstrained* space,
  where a ``real<lower=0>`` parameter becomes ``exp(u)`` and the log density
  includes the log-Jacobian ``u``.  The experiments therefore use that
  parameterization by default.  Comparison with the HMC summaries happens in the
  constrained space: for the exponential coordinates the variational Gaussian on
  ``u`` induces lognormal moments, which `PosteriorTarget.variational_summaries`
  computes exactly.
* **Validation.**  `tests/test_targets.py` checks ``E_p[grad log p(z)] = 0`` under
  the HMC reference draws, in both parameterizations, for all three models.
  This is a strong check that the transposed Stan models, the data, the
  parameter order and the constraint handling all match the reference posterior.
* **Reference draws.**  The `arK` and `eight_schools` draws map directly onto the
  model parameters.  For `gp_pois_regr`, posteriordb stores the *transformed*
  parameter ``f``; since the model's parameters are ``(rho, alpha, f_tilde)``
  with ``f = L(rho, alpha) f_tilde``, the draws are mapped back with
  ``f_tilde = L(rho, alpha)^{-1} f`` in `experiments/prepare_posteriordb.py`.
  `eight_schools_centered` has no draw file of its own; posteriordb's non-centered
  posterior is the same distribution, and its draws contain the transformed
  parameter ``theta`` (= ``mu + tau * theta_tilde``), so the reference summaries
  are exact.

## 3. Baselines (Appendix E.1)

* ADVI maximizes the ELBO with Adam, using the reparameterization trick
  (Algorithm 2).  The variational covariance is parameterized by its Cholesky
  factor with an exponential diagonal (so that all positive-definite covariances
  are reachable and the optimizer is unconstrained).
* "Score" replaces the ELBO by the score-based divergence of eq. (2), i.e. with
  the ``Cov(q)``-weighted norm as in eq. (93); "Fisher" uses the unweighted norm.
  Both use the same Adam optimizer.
* GSM follows Algorithm 3 exactly, with an eigenvalue clip applied to the
  averaged covariance update so that it stays positive definite (the paper notes
  that GSM's updates can push the covariance outside the PSD cone when the
  target is highly non-Gaussian; the clip is set at ``1e-12``).

## 4. Learning-rate grid searches

The addendum says that a grid search was used for the gradient-based methods in
Sections 5.1, 5.2 and 5.3.  `src/bam/grid_search.py` implements the search and
records both the candidate grids and the values selected in the paper
(``PAPER_SELECTED``).  The experiment scripts use the paper's values by default
and re-derive them with ``--grid-search``.  Note that a grid search performed
with a shorter pilot run than the paper's can select a larger learning rate.

## 5. Deep generative model (Section 5.3, Appendix E.6)

* The encoder/decoder architecture follows the addendum exactly (5 convolutional
  layers with GELU activations, no normalization/dropout, ``stride = 2``
  downsampling, ``tanh`` output, latent dimension 256).  The hidden width
  ``c_hid`` is *not* specified in the paper; the default is 64 (``--c-hid``).
* The addendum lists a mean head for the encoder; the implementation adds a
  log-variance head, which amortized variational inference (AVI) requires.
* CIFAR-10 images are rescaled to ``[-1, 1]`` to match the ``tanh`` output, and
  the likelihood variance is ``sigma^2 = 0.1`` as in the paper.
* Training uses the addendum's optimizer settings (Adam, linear warmup 0 -> 1e-4
  over 100 batches, linear decay to 1e-5 over 500 batches) and one Monte-Carlo
  sample for the negative ELBO.
* AVI uses the encoder mean as the variational mean and the encoder variance for
  the SD (the paper's AVI baseline is a factorized Gaussian).

## 6. Runtime guidance

The defaults of the experiment scripts mirror the paper; the full runs are long
and were **not** executed in the development environment (no GPU).

| Script | Default configuration | Rough cost |
| --- | --- | --- |
| `run_gaussian_targets.py` | D = 4, 16, 64, 256; 10 runs; up to 1e5 gradient evaluations; 7 methods | hours on CPU; the D = 256 runs dominate |
| `run_shash_targets.py` | 6 targets; 10 runs; 9 methods; Monte-Carlo KL with 2e4 samples per point | tens of minutes on CPU |
| `run_posteriordb.py` | 3 models; 2 batch sizes; 5 runs; BaM/ADVI/GSM | minutes on CPU |
| `train_vae_cifar10.py` | 100 epochs of CIFAR-10 at ``c_hid = 64`` | GPU recommended (hours on CPU) |
| `run_vae_posterior.py` | B = 10, 100, 300; pilot 100 + main 1000 iterations | GPU recommended |
| `verify_theorem31.py` | D = 16, 200 iterations | seconds |

Useful flags for cheaper runs: ``--quick`` (all experiment scripts),
``--max-grad-evals``, ``--n-runs``, ``--eval-points``, ``--n-eval-samples``,
``--dims/--bam-batches`` (Gaussian targets).

## 7. Explicitly out of scope

Following the addendum, the following are not part of the reproduction:

* wallclock-timing figures (E.1, and the wallclock panel E.7);
* the BaM learning-rate schedule sweeps for Gaussian and non-Gaussian targets
  (Figures E.2 and E.5);
* any experiment introduced only in the appendix.

The grid searches for the gradient-based methods remain in scope and are
implemented, as are the relative-SD-error results of Figure E.6 (which the
addendum explicitly keeps in scope).
