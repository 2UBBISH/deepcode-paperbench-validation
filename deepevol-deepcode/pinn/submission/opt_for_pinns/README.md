# Challenges in Training PINNs: A Loss Landscape Perspective — Reproduction

Code reproduction of **"Challenges in Training PINNs: A Loss Landscape Perspective"**
(ICML 2024, PMLR 235).

The repository implements:

1. **PINN infrastructure** for three PDE benchmarks (convection, reaction, wave) —
   residual/BC operators, sampling protocol, loss (Eq. 2) and the L2RE metric (Eq. 3).
2. **Optimizer comparison** — Adam / L-BFGS / Adam+L-BFGS (Fig. 2, Fig. 8, Table 1).
3. **Spectral-density analysis** of the loss Hessian `H_L(w)` and of its
   L-BFGS-preconditioned form `H̃_kᵀ H_L(w) H̃_k` (Fig. 3, Fig. 7).
4. **NysNewton-CG (NNCG)** second-order fine-tuning that continues to decrease the loss
   after Adam+L-BFGS stalls (Fig. 1, Fig. 4, Fig. 5, Table 2, Table 3).

§6.2, §8 and all theoretical results are out of scope per the reproduction addendum.

---

## 1. Environment

| Component | Version |
|---|---|
| Python | 3.10.12 |
| PyTorch | 2.0.0 (CUDA 11.8 build recommended) |
| NumPy | `>=1.23,<2.0` |
| SciPy | `>=1.9.0` (QR / Cholesky / SVD in the Nyström routines) |
| PyHessian | optional SLQ backend |
| matplotlib / pyyaml / tqdm / pandas | figures, config, progress, tables |
| pytest | test suite |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GPU build of PyTorch 2.0.0:
# pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

**Precision:** all second-order routines (HVP, Lanczos/SLQ, Nyström, NyströmPCG, NNCG)
run in `float64` (`runtime.dtype: float64` in `configs/default.yaml`). A GPU is
recommended for the wave PDE (NNCG is HVP-heavy), but the pipeline runs on CPU.

---

## 2. Repository layout

```
opt_for_pinns/
├── run.py                              # CLI dispatcher: compare | spectral | nncg | all
├── configs/default.yaml                # PDE coefficients, hyperparameters, sweep axes
├── src/
│   ├── pinns/
│   │   ├── model.py                    # MLP u(x;w): tanh, 3 hidden layers, Xavier+zero-bias
│   │   ├── problems.py                 # Convection / Reaction / Wave: D, B, exact solution
│   │   ├── sampling.py                 # 10k residual pts from 255×100 grid; 257 IC; 101 BC
│   │   ├── loss.py                     # PINN loss (Eq. 2) + residual/IC/BC components
│   │   └── metrics.py                  # L2RE (Eq. 3) + full-grid / per-region evaluation
│   ├── optimizers/
│   │   ├── first_order.py              # Adam (LR grid) + GD control + training driver
│   │   ├── lbfgs_wrapper.py            # L-BFGS (lr=1, m=100, strong Wolfe) + (s,y,ρ) recording
│   │   ├── combined.py                 # Adam+L-BFGS with switch iteration
│   │   ├── armijo.py                   # Algorithm 7 — Armijo backtracking line search
│   │   ├── nystrom.py                  # Algorithm 5 (RandomizedNyström), Algorithm 6 (NyströmPCG)
│   │   └── nncg.py                     # Algorithm 4 — NysNewton-CG optimizer
│   ├── spectral/
│   │   ├── hvp.py                      # Pearlmutter Hessian-vector products + flat-param utils
│   │   ├── lbfgs_unroll.py             # Algorithm 2 — unrolled L-BFGS (Ỹ, Ṽ, S̃)
│   │   ├── preconditioned_mvp.py       # Algorithm 3 — H̃ᵀ H_L H̃ mat-vec
│   │   └── spectral_density.py         # SLQ / (optional) PyHessian spectral densities
│   └── utils/
│       ├── seeding.py                  # Reproducible RNG control
│       └── plotting.py                 # Figure helpers (Fig. 1–8)
├── experiments/
│   ├── run_optimizer_comparison.py     # Fig 2, Fig 8, Table 1
│   ├── run_spectral_density.py         # Fig 3, Fig 7
│   └── run_nncg_finetune.py            # Fig 1, Fig 4, Fig 5, Table 2, Table 3
├── tests/test_problems.py              # analytical-solution + loss==0 + HVP sanity checks
├── requirements.txt
└── README.md
```

---

## 3. Paper specification implemented

### 3.1 PDE problems (§2.1, Appendix A)

| Problem | Operator `D` | Coefficient | Domain | Boundary / Initial | Exact solution |
|---|---|---|---|---|---|
| Convection | `u_t + β u_x` | `β = 40` | `(0,2π)×(0,1)` | periodic BC; `u(x,0)=sin x` | `u = sin(x − βt)` |
| Reaction | `u_t − ρ u(1−u)` | `ρ = 5` | `(0,2π)×(0,1)` | periodic BC; Gaussian IC `exp(−(x−π)²/(2(π/4)²))` | `u = h e^{ρt} / (h e^{ρt} + 1 − h)` |
| Wave | `u_tt − 4 u_xx` | `β = 5` | `(0,1)×(0,1)` | `u(0,t)=u(1,t)=0`; `u(x,0)=sin(πx)+0.5 sin(βπx)`, `u_t(x,0)=0` | two-mode standing wave |

Residuals are built with `torch.autograd.grad(..., create_graph=True)` so the loss stays
twice-differentiable w.r.t. the parameters.

### 3.2 Network and sampling (§2.2)

* MLP `u(x;w): R² → R`, `tanh` activations, **3 hidden layers**, width ∈ {50, 100, 200, 400},
  Xavier-normal weights, **all biases initialised to 0**.
* Sampling: **10 000** residual collocation points drawn uniformly at random from the
  **255×100** interior grid, **257** equally-spaced initial-condition points, **101**
  equally-spaced points per boundary.
* Loss (Eq. 2):
  `L(w) = 1/(2 n_res) Σ (D[u(x_r;w)])² + 1/(2 n_bc) Σ (B[u(x_b;w)])²`,
  exposed together with its residual / initial / boundary components.
* Metric (Eq. 3): `L2RE = ‖y − y′‖₂ / ‖y′‖₂` over the full grid + IC/BC points,
  `y′` from the analytical solutions above.

### 3.3 Optimizers (§2.2, §6.1)

* **Adam** with LR grid `{1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`, tuned per width.
* **L-BFGS** with `lr = 1.0`, memory `m = 100`, strong-Wolfe line search, recording the
  `(s_k, y_k, ρ_k, γ_k)` curvature buffers.
* **Adam+L-BFGS** with switch at 1 k / 11 k / 31 k iterations; total budget
  **41 000 iterations**.

### 3.4 Spectral-density pipeline (§5.1–5.3, Appendix C.2)

* **Algorithm 2** (`lbfgs_unroll.py`): unrolls the recorded L-BFGS buffers into `Ỹ, Ṽ, S̃`
  so that `H_k = H̃_k H̃_kᵀ` can be applied without materialising `H_k`.
* **Algorithm 3** (`preconditioned_mvp.py`): computes matrix-vector products with
  `H̃_kᵀ H_L(w) H̃_k` (split `v = (v₁, v₂)`, apply HVP, restack).
* SLQ spectral densities (`spectral_density.py`) for `H_L`, for the preconditioned
  operator, and per loss component (residual / initial / boundary), reporting outlier
  eigenvalues and the condition-number reduction.

### 3.5 NNCG (§7.2, Appendix E.2)

* **Algorithm 5** `RandomizedNyströmApproximation`: top-`s` approximate eigenpairs via a
  Gaussian sketch `Y = M Q`; `ν = √p · ε·‖Y‖₂`; indefinite fail-safe branch (eigendecomposition
  + shift) when the Cholesky of `QᵀY_ν` fails; `Λ̂ = max(0, Σ² − (ν + |λ|))`.
* **Algorithm 6** `NyströmPCG`: preconditioned CG for `(A + μI)x = b`, warm-started at
  `x₀`, with `P = 1/(λ̂_s+μ) U(Λ̂+μI)Uᵀ + (I − UUᵀ)` and its inverse.
* **Algorithm 7** Armijo backtracking (`α = 0.1`, `β = 0.5`).
* **Algorithm 4** NNCG: refresh `(U, Λ̂)` every `F` iterations, damped Newton step via
  NyströmPCG, `η_k` from Armijo, `w_{k+1} = w_k − η_k d_k`.
  Defaults: `η=1, K=2000, s=60, F=20, ε=1e-16, M=1000, α=0.1, β=0.5`,
  `μ ∈ {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}` (tuned; `1e-2`/`1e-1` usually best).

---

## 4. Running the experiments

```bash
# 0. correctness gate (analytic residuals < 1e-10, loss == 0 at the exact solution, HVP vs FD)
python tests/test_problems.py
# or: pytest -q tests/test_problems.py

# 1. optimizer comparison  -> results/optimizer_comparison/  (Fig 2, Fig 8, Table 1)
python run.py compare

# 2. spectral densities    -> results/spectral_density/      (Fig 3, Fig 7)
python run.py spectral

# 3. NNCG fine-tuning      -> results/nncg_finetune/         (Fig 1, Fig 4, Fig 5, Tab 2–3)
python run.py nncg

# everything in order
python run.py all

# fast smoke test (1 width, 1 seed, reduced budgets)
python run.py all --quick

# cluster example: 4 GPUs, 2 jobs each, separate output dirs
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python run.py compare --outdir results/gpu$g --quiet &
done
```

Useful shared flags: `--config PATH`, `--pdes convection,reaction,wave`,
`--widths 50,100,200,400`, `--lrs 1e-4,1e-3,1e-2`, `--seeds 345,456,567`,
`--optimizers adam,lbfgs,adam+lbfgs`, `--switch 11000`, `--total-iterations 41000`,
`--finetune-steps 2000`, `--device cpu|cuda`, `--no-plots`, `--quiet`.

Individual experiment scripts can also be invoked directly
(`python experiments/run_spectral_density.py --quick`).

### Outputs

```
results/
├── environment.json                     # python/torch/cuda + module availability snapshot
├── optimizer_comparison/
│   ├── summary.json  table1.json  table1.txt  progress.json
│   ├── checkpoints/*.pt                 # model state + recorded L-BFGS buffers
│   └── figures/figure2_*, figure8_*
├── spectral_density/
│   ├── summary.json  checkpoints/*.pt
│   └── figures/figure3_*, figure7_*
└── nncg_finetune/
    ├── summary.json  table2.json/txt  table3.json/txt
    ├── *_absolute_errors.npz
    └── figures/figure1_*, figure4_*, figure5_*
```

Each run stores the trained `state_dict` together with the L-BFGS curvature history, so the
spectral-density and NNCG experiments can be re-run without retraining.

---

## 5. Best-configuration selection rule

The paper does **not** hard-code the winning hyper-parameters; they are re-derived by the
same systematic process used in the paper:

> For each PDE, train Adam+L-BFGS with an **11 000-iteration switch point** over the grid
> `width ∈ {50, 100, 200, 400} × Adam lr` (tuned) `× seed ∈ {345, 456, 567, 678, 789}`,
> and select the run with the **smallest L2RE** (ties broken by loss, then canonical order).
> The spectral-density and NNCG experiments start from that run.

This is encoded in `configs/default.yaml`:

```yaml
experiment:
  selection: l2re              # smallest L2RE
  selection_keys: [pde, width, adam_lr, seed]
  seeds: [345, 456, 567, 678, 789]
  switch_iteration: 11000
```

In the paper this process selected width 200 and seeds 345/456/567 with Adam lr
1e-4/1e-3/1e-3. A faithful reproduction must follow the same rule; matching the paper's
exact winners is not required (addendum §E).

Adam learning rates for the optimizer comparison are tuned per width by **lowest loss**
(`optimizer.lr_selection: loss`), matching §6.1.

---

## 6. Reproducibility and seeds

* All RNGs (Python `random`, NumPy, torch CPU/CUDA) are seeded through
  `src/utils/seeding.set_seed` before model creation, sampling, and each training run.
* The same residual collocation points are used across optimizers for a given
  `(pde, seed)` pair, making the comparison fair.
* `float64` everywhere for the second-order path.
* Histories written to `summary.json` are downsampled (≤256 points) for compactness;
  full curves remain available through the `--json-summary` payload if requested.

---

## 7. Expected results (paper reference)

### Table 1 — optimizer comparison (lowest values over widths/seeds)

| PDE | | Adam | L-BFGS | Adam+L-BFGS |
|---|---|---|---|---|
| Convection | loss | 1.40e-4 | 1.51e-5 | **5.95e-6** |
| | L2RE | 5.96e-2 | 8.26e-3 | **4.19e-3** |
| Reaction | loss | 4.73e-6 | 8.93e-6 | **3.26e-6** |
| | L2RE | 2.12e-2 | 3.83e-2 | **1.92e-2** |
| Wave | loss | 2.03e-2 | 1.84e-2 | **1.12e-3** |
| | L2RE | 3.49e-1 | 3.35e-1 | **5.52e-2** |

Qualitative criterion: Adam+L-BFGS attains lower minimum loss **and** lower minimum L2RE
than Adam or L-BFGS alone at most widths (reaction may be an exception), and Fig. 2 shows
lower loss ↔ lower L2RE.

### Spectral densities (Fig. 3, Fig. 7)

* `H_L` has large outlier eigenvalues and a mass of eigenvalues near 0:
  outliers `> 1e4` (convection), `> 1e3` (reaction), `> 1e5` (wave).
* The **residual** component is the most ill-conditioned.
* L-BFGS preconditioning reduces the top eigenvalue and the condition number by **at least
  10³**; every loss component is improved by the preconditioner.

### Table 2 — NNCG / GD fine-tuning after Adam+L-BFGS

| PDE | Adam+L-BFGS (loss / L2RE) | after NNCG | after GD (control) |
|---|---|---|---|
| Convection | 5.95e-6 / 4.19e-3 | **3.63e-6 / 1.94e-3** | unchanged |
| Reaction | 5.26e-6 / 1.92e-2 | **2.89e-7 / 9.92e-3** | unchanged |
| Wave | 1.12e-3 / 5.52e-2 | **6.13e-5 / 1.27e-2** | unchanged |

Fig. 4: NNCG reduces the loss by more than 10× and lowers the gradient norm on convection
and wave, whereas GD makes no progress. Fig. 1: on the wave PDE, NNCG continues to reduce
the loss after L-BFGS stalls near 40 k iterations.

### Table 3 — per-iteration wall-clock ratio NNCG / L-BFGS

| Convection | Reaction | Wave |
|---|---|---|
| ≈ 5.43 | ≈ 20 | ≈ 322 |

(Ratios depend on hardware; the qualitative ordering NNCG ≫ L-BFGS on the wave PDE holds.)

---

## 8. Design notes / resolution of ambiguities

Where the paper is silent, the following documented defaults were chosen:

* **Input/output dims**: `in_dim = 2` (space, time), `out_dim = 1` for all three PDEs.
* **Residual sampling**: uniform random interior points drawn from the flattened 255×100
  grid (with replacement); IC/BC points equally spaced along the free axis, periodic
  conditions contributing both ends of the fixed axis.
* **GD fine-tuning LR** (control experiment only): fixed at `1e-4`
  (`optimizer.gd.lr`).
* **`K = 2000`** denotes **2000 additional** fine-tuning steps after Adam+L-BFGS.
* **SLQ settings**: `n_iter = 100` Lanczos iterations, `n_vec = 1` Rademacher probe,
  `200`-point density grid, `backend: native` (PyHessian optional).
* **Failure of a component** degrades gracefully: training wraps every stage in
  `try/except`, optional imports (PyHessian, matplotlib, PyYAML, numpy) are guarded, and
  NNCG falls back NyströmPCG → plain CG → gradient direction.
* Where a run is borderline (e.g. reaction, where Adam occasionally matches
  Adam+L-BFGS), the systematic selection process is followed rather than chasing exact
  paper numbers.

---

## 9. Module map (paper → code)

| Paper component | File |
|---|---|
| MLP ansatz (§2.2, App. A) | `src/pinns/model.py` |
| PDE operators `D`, `B`, exact solutions (App. A.1–A.3) | `src/pinns/problems.py` |
| Sampling protocol (§2.2) | `src/pinns/sampling.py` |
| Loss, Eq. (2) + components (§2.1) | `src/pinns/loss.py` |
| L2RE, Eq. (3) (§2.2) | `src/pinns/metrics.py` |
| Adam / GD + LR grid (§2.2, §6.1) | `src/optimizers/first_order.py` |
| L-BFGS + curvature recording (§2.2, §6.1) | `src/optimizers/lbfgs_wrapper.py` |
| Adam+L-BFGS switching (§6.1) | `src/optimizers/combined.py` |
| Armijo line search, Alg. 7 (App. E.2) | `src/optimizers/armijo.py` |
| RandomizedNyström Alg. 5, NyströmPCG Alg. 6 (App. E.2) | `src/optimizers/nystrom.py` |
| NNCG, Alg. 4 (§7.2, App. E.2) | `src/optimizers/nncg.py` |
| Hessian-vector products (§7.2) | `src/spectral/hvp.py` |
| Unrolled L-BFGS, Alg. 2 (App. C.2) | `src/spectral/lbfgs_unroll.py` |
| Preconditioned mat-vec, Alg. 3 (App. C.2) | `src/spectral/preconditioned_mvp.py` |
| SLQ spectral densities (§5.1–5.3) | `src/spectral/spectral_density.py` |
| Fig. 2 / 8 / Table 1 | `experiments/run_optimizer_comparison.py` |
| Fig. 3 / 7 | `experiments/run_spectral_density.py` |
| Fig. 1 / 4 / 5 / Table 2 / 3 | `experiments/run_nncg_finetune.py` |
| Config + dispatch | `configs/default.yaml`, `run.py` |
| Correctness tests | `tests/test_problems.py` |

---

## 10. Citation

```bibtex
@inproceedings{rathore2024challenges,
  title     = {Challenges in Training {PINN}s: A Loss Landscape Perspective},
  author    = {Rathore, Pratik and Lei, Weimu and Frangella, Zachary and Lu, Lu and
               Udell, Madeleine},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {PMLR},
  volume    = {235},
  year      = {2024}
}
```
