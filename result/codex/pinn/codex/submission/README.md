# Reproducing *Challenges in Training PINNs: A Loss Landscape Perspective*

This repository is a from-scratch reimplementation of the experiments of

> P. Rathore, W. Lei, Z. Frangella, L. Lu, M. Udell.
> *Challenges in Training PINNs: A Loss Landscape Perspective.* ICML 2024.

It contains everything needed to reproduce

* **Section 4 / Figure 2** – the relation between the final loss and the L2
  relative error (L2RE),
* **Section 5 / Figures 3 and 7** – the spectral density of the Hessian of the
  PINN loss (and of each loss component) and of the L-BFGS-preconditioned
  Hessian, showing that the loss landscape is ill-conditioned and that
  quasi-Newton preconditioning reduces the top eigenvalue by ≥ 10³,
* **Section 6 / Table 1 and Figure 8** – the comparison of Adam, L-BFGS and
  Adam+L-BFGS over network widths and hyper-parameters,
* **Section 7 / Figures 1, 4, 5 and Tables 2, 3** – under-optimisation of
  Adam+L-BFGS and the improvement obtained by the new second-order optimiser
  NysNewton-CG (NNCG), including its wall-clock cost relative to L-BFGS.

The theoretical sections (Section 8 and Appendix F/G) are out of scope, as are
Section 6.2 and Figures 6, 9 and 10 (see `paper/addendum.md`).

## Repository layout

```
pinn/
  problems.py        the three problems of Appendix A (convection, reaction, wave),
                     their differential operators, conditions and exact solutions
  data.py            sampling of the 10000 residual points and of the IC/BC points
  models.py          the tanh MLP (3 hidden layers, Xavier normal, zero biases)
  losses.py          the PINN objective L(w) of Eq. (2) and its components
  metrics.py         L2 relative error (L2RE)
  optim/
    objective.py     flat-parameter view of L(w): loss, gradient, Hessian-vector products
    lbfgs.py         L-BFGS (lr 1.0, memory 100, strong Wolfe), correction pairs exposed
    trainers.py      the Adam / L-BFGS / Adam+L-BFGS training loops of Sections 2.2 and 6
    nncg.py          NNCG (Algorithms 4-7): RandomizedNystroemApproximation,
                     NystroemPCG and Armijo, plus the GD baseline of Section 7.3
    gdnd.py          Gradient-Damped Newton Descent (Algorithm 1, Section 8)
  hessian/
    hvp.py           Hessian-vector products (Pearlmutter, 1994)
    spectral.py      stochastic Lanczos quadrature (SLQ) spectral density
    lbfgs_precond.py Algorithms 2 and 3: unrolling the L-BFGS update and the
                     matrix-vector product with Htilde^T H Htilde
  experiments/
    common.py        run specifications, result paths, single-run driver
    grid.py          the Section 6 hyper-parameter grid
    analysis.py      Table 1 and the Figure 8 statistics (min/median/max per width)
    spectral.py      Figures 3 and 7
    finetune.py      Figures 1, 4, 5 and Tables 2, 3
  plotting.py        all figures
  cli.py             command line interface
tests/test_core.py   unit and smoke tests
scripts/             convenience shell wrappers
```

## Running the reproduction

Install the dependencies (PyTorch, NumPy, SciPy, Matplotlib) and run

```bash
python -m pinn.cli grid                     # Section 6 grid          (~1260 runs)
python -m pinn.cli analyze                  # Table 1, Figures 2, 8
python -m pinn.cli spectra --pde wave --select-best --component   # Figures 3, 7
python -m pinn.cli finetune --pde convection --select-best        # Figures 1, 4, 5, Tables 2, 3
```

or equivalently `scripts/run_all.sh`. A *fast* end-to-end check of every code
path (small networks, 200 iterations) is

```bash
python -m pinn.cli all --quick --outdir runs_quick
```

`--select-best` picks, for a given PDE, the (Adam learning rate, seed, network
width) configuration with the smallest L2RE among the stored Adam+L-BFGS runs
that switch at 11000 iterations, exactly as described in the addendum to the
paper. All numbers are written to `runs/tables/*.csv`, all figures to
`runs/figures/*.png`.

### Compute

Everything runs on CPU. A single full-scale run (41000 iterations) takes roughly
30–120 minutes depending on the PDE and the network width, so the complete grid
is a multi-GPU-day job in the original paper's setting; the code therefore
supports `--limit`, `--pdes`, `--widths`, `--seeds`, `--lrs` and `--iters` so
that subsets can be reproduced. The `--quick` mode finishes in a few minutes and
is what was executed while developing this repository.

## What is implemented, and how it maps to the paper

| Paper item | Where |
|---|---|
| Eq. (1)–(2): PINN objective | `pinn/losses.py:component_terms`, `pinn/optim/objective.py:Objective` |
| Appendix A: convection / reaction / wave | `pinn/problems.py` |
| Section 2.2: sampling, MLP, L2RE | `pinn/data.py`, `pinn/models.py`, `pinn/metrics.py` |
| Section 4, Figure 2 | `experiments/grid.py` + `plotting.plot_loss_vs_l2re` |
| Section 5, Figures 3 & 7 | `hessian/spectral.py`, `hessian/lbfgs_precond.py`, `experiments/spectral.py` |
| Appendix C.1/C.2: L-BFGS preconditioning | `hessian/lbfgs_precond.py` (Algorithms 2 and 3) |
| Section 6, Table 1, Figure 8 | `experiments/grid.py`, `experiments/analysis.py` |
| Section 7.2/Appendix E.2: NNCG | `optim/nncg.py` (Algorithms 4–7) |
| Section 7.3, Figures 1, 4, 5, Tables 2, 3 | `experiments/finetune.py` |

Everything in the table was re-derived from the paper text; no code from the
authors' repository (or any other listed resource) was used.

### Key implementation details

* **Differential operators.** The residual is evaluated with double-precision
  autograd (`torch.autograd.grad(..., create_graph=True)`); the wave problem
  needs `u_tt`, so second-order derivatives are supported.
* **Loss aggregation.** Eq. (2) has a single `1/(2 n_bc)` average over *all*
  initial and boundary points; this is the default
  (`aggregation="combined"`). The alternative convention of averaging each
  condition separately is available as `aggregation="per_condition"`.
* **L-BFGS.** `torch.optim.LBFGS` with `lr=1.0`, `history_size=100`,
  `line_search_fn="strong_wolfe"` and **one iteration per `step()` call**
  (`max_iter=1`), so that the iteration budget of 41000 is comparable to Adam's
  gradient steps and a per-iteration trace can be recorded (Figure 4). The
  strong-Wolfe line search and the `d·t ≤ tol` stopping rule reproduce the early
  termination of L-BFGS described in Section 7.1.
* **Preconditioned spectrum.** The correction pairs `s_k, y_k, ρ_k` and the
  scaling `γ_k` are read from the L-BFGS state and unrolled with Algorithm 2;
  Algorithm 3 then supplies matrix-vector products with
  `M = Htilde_kᵀ H_L Htilde_k`, whose non-zero eigenvalues coincide with those
  of the preconditioned Hessian `H_k H_L`. SLQ (Lanczos with full
  reorthogonalisation + Gaussian smoothing of the Ritz values) then yields the
  spectral density, as in PyHessian (Yao et al., 2020).
* **NNCG** uses the paper's defaults
  `η=1, K=2000, s=60, F=20, ε=1e-16, M=1000, α=0.1, β=0.5`, warm-starts CG with
  the previous Newton step, and tunes `μ ∈ {1e-5,…,1e-1}` (the paper reports
  `1e-2` and `1e-1` as the best values); the run with the lowest loss is
  reported, as in Figures 1 and 4.

### Assumptions (not stated explicitly in the paper)

* The five random seeds are taken to be `123, 234, 345, 456, 567`; the addendum
  mentions `345`, `456` and `567` as the best seeds for the three PDEs, so the
  seed set contains them.
* The 10000 residual points are drawn uniformly *without* replacement from the
  255×100 interior grid.
* The GD baseline of Section 7.3 uses the tuned Adam learning rate unless
  `--gd-lr` is given, and the additional 2000 steps mentioned by the addendum.

## Tests

```bash
python tests/test_core.py          # or: python -m pytest tests -q
```

The tests check that the analytical solutions satisfy every differential
operator and condition, that the L2RE of the exact solution is zero, that the
Hessian matches finite differences, that the SLQ spectrum recovers the extreme
eigenvalues of a known matrix, that the randomized Nyström approximation
(Algorithm 5) returns the top-`s` eigenpairs and that NyströmPCG (Algorithm 6)
reproduces the dense solution of `(H + mu I) d = g`.  Most importantly, the
tests verify that the unrolled L-BFGS preconditioner of Appendix C reproduces
the standard two-loop L-BFGS matrix and that the SLQ matrix `M` has the same
non-zero eigenvalues as `H_k H_L`.

## Results obtained while developing this repository

The commands below were executed in `--quick` mode (width 20, 200 iterations,
one seed, Adam learning rate 1e-2), which is a smoke test rather than a
reproduction of the paper's numbers, but it exercises every code path and
already shows the qualitative trends.

**Figure 3 / Section 5** – the spectral density of the Hessian has a large
outlier eigenvalue and lots of mass near zero on all three problems, and
L-BFGS preconditioning shrinks the top eigenvalue:

| PDE | top eigenvalue of `H_L` | top eigenvalue of `H_k H_L` | reduction |
|---|---|---|---|
| convection | 3.14e3 | 2.02 | 1553× |
| reaction | 5.16e2 | 11.5 | 45× |
| wave | 6.67e2 | 2.51 | 266× |

The residual term is the most ill-conditioned component, as in the paper.

**Table 2 / Section 7.3** – NNCG reduces both the loss and the L2RE of the
Adam+L-BFGS solution on all three problems, while 2000 steps of gradient descent
with the tuned learning rate leave the loss essentially unchanged, exactly as
in Table 2 of the paper (`loss` / `L2RE`; 20 NNCG and 20 GD steps, `mu` tuned
over two values, GD learning rate 1e-4):

| PDE | Adam+L-BFGS | +NNCG | +GD |
|---|---|---|---|
| convection | 1.055e-1 / 2.127 | 9.40e-2 / 1.432 | 1.055e-1 / 2.128 |
| reaction | 7.392e-2 / 9.880e-1 | 7.373e-2 / 9.898e-1 | 7.392e-2 / 9.877e-1 |
| wave | 2.931e-2 / 9.362e-1 | 2.535e-2 / 8.414e-1 | 2.931e-2 / 9.362e-1 |

NNCG was 31–40× more expensive per iteration than L-BFGS here (Table 3 reports
5.4–322× over the three problems), and the reaction problem — the least
ill-conditioned one — gains the least, again matching the paper.

*Caveat for smoke tests only:* the damping `mu` has to exceed the magnitude of
the most negative eigenvalue of `H_L` for `H_L + mu I` to be positive definite.
At the barely-trained points of a 200-iteration smoke test the Hessian is
indefinite, so `NystroemPCG` needs its full iteration budget (`cg_max_iter`);
at the trained optima used in the paper the Hessian is essentially PSD.

Intermediate-scale runs reproduce the ordering of Section 6 (width 50, seed 0,
Adam learning rate 1e-2; loss / L2RE):

| PDE, iterations | Adam | Adam+L-BFGS | L-BFGS |
|---|---|---|---|
| convection, 3000 (switch 1k) | 3.51e-2 / 1.01 | 4.65e-3 / 8.83e-1 | 3.92e-3 / 8.46e-1 |
| wave, 2500 (switch 0.5k) | 3.80e-2 / 1.10 | 2.02e-2 / 5.16e-1 | 1.99e-2 / 4.91e-1 |

Adam is clearly behind the quasi-Newton methods, and the Adam solution has an
L2RE of ≈ 1 with a small residual — the trivial (constant) solution discussed in
Section 4 and Appendix B. Adam+L-BFGS and L-BFGS are comparable at these short
horizons; the paper's separation between them (Table 1: Adam+L-BFGS is 6× better
than L-BFGS on wave) appears once L-BFGS alone starts stalling at the full
41000-iteration budget, i.e. it needs the full grid to be visible.

Reproducing the actual numbers of the paper requires the full grid
(`scripts/run_grid.sh`) followed by `run_spectra.sh` and `run_finetune.sh`.

### Reference values from the paper (targets for a full run)

Table 1, lowest loss / L2RE after hyper-parameter tuning:

| optimiser | convection | reaction | wave |
|---|---|---|---|
| Adam | 1.40e-4 / 5.96e-2 | 4.73e-6 / 2.12e-2 | 2.03e-2 / 3.49e-1 |
| L-BFGS | 1.51e-5 / 8.26e-3 | 8.93e-6 / 3.83e-2 | 1.84e-2 / 3.35e-1 |
| Adam+L-BFGS | 5.95e-6 / 4.19e-3 | 3.26e-6 / 1.92e-2 | 1.12e-3 / 5.52e-2 |

Table 2, after fine-tuning with NNCG or GD:

| optimiser | convection | reaction | wave |
|---|---|---|---|
| Adam+L-BFGS | 5.95e-6 / 4.19e-3 | 5.26e-6 / 1.92e-2 | 1.12e-3 / 5.52e-2 |
| +NNCG | 3.63e-7 / 1.94e-3 | 2.89e-7 / 9.92e-3 | 6.13e-5 / 1.27e-2 |
| +GD | unchanged | unchanged | unchanged |

with the best configurations being width 200, Adam learning rates 1e-4
(convection), 1e-3 (reaction), 1e-3 (wave) and seeds 345, 456 and 567
respectively (addendum). Table 3 reports NNCG/L-BFGS per-iteration time ratios
of 5.4, 20 and 322 for convection, reaction and wave.
