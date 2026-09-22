# Challenges in Training PINNs: A Loss Landscape Perspective

Reference implementation of the experiments in *"Challenges in Training PINNs: A Loss
Landscape Perspective"*.

This codebase reproduces the paper's central claims:

1. **The PINN loss is ill-conditioned.** The differential operators in the PDE residual
   term produce Hessian spectra with very large outlier eigenvalues and a large mass of
   eigenvalues near zero (Figures 3 & 7).
2. **Adam + L-BFGS is superior to Adam or L-BFGS alone.** Switching from Adam to L-BFGS
   yields the best final loss / L2 relative error across PDEs and widths (Table 1,
   Figures 2 & 8).
3. **NysNewton-CG (NNCG) further improves PINN solutions.** A second-order optimizer
   built on a randomized Nyström Hessian approximation + preconditioned CG + Armijo line
   search reduces the loss by more than 10x relative to Adam + L-BFGS (Table 2,
   Figures 1, 4, 5).

---

## 1. Repository layout

```
opt_for_pinns/
├── main.py                          # CLI: train / eval / spectral
├── run_experiments.py               # Full experiment grid (PDE x optimizer x width x seed)
├── src/
│   ├── pdes.py                      # Convection, Reaction, Wave PDEs + analytical solutions
│   ├── model.py                     # MLP (tanh, 3 hidden layers, Xavier init, zero biases)
│   ├── loss.py                      # PINN loss (Eq. 2) + per-component losses
│   ├── data.py                      # Grid generation & sampling (10000 res, 257 IC, 101 BC)
│   ├── metrics.py                   # L2RE (Eq. 3), gradient norm, condition number
│   ├── utils.py                     # Seeding, logging, checkpointing, config loading
│   ├── optimizers/
│   │   ├── adam_lbfgs.py            # Adam+L-BFGS switching wrapper (1k/11k/31k)
│   │   ├── nncg.py                  # NysNewton-CG (Algorithm 4)
│   │   ├── nystrom.py               # RandomizedNystromApproximation (Algorithm 5)
│   │   ├── pcg.py                   # NystromPCG (Algorithm 6) + preconditioner P, P^{-1}
│   │   └── armijo.py                # Armijo line search (Algorithm 7)
│   └── hessian/
│       ├── hvp.py                   # Hessian-vector products (Pearlmutter trick)
│       ├── spectral_density.py      # SLQ-based spectral density (PyHessian-style)
│       ├── lbfgs_unroll.py          # Unrolling L-BFGS update (Algorithm 2)
│       └── precond_matvec.py        # Preconditioned Hessian matvec (Algorithm 3)
├── configs/
│   ├── convection.yaml              # beta=40, domain, lr grid, best config
│   ├── reaction.yaml                # rho=5
│   └── wave.yaml                    # beta=5
├── scripts/
│   ├── reproduce_table1.py          # Table 1: best loss/L2RE per optimizer
│   ├── reproduce_fig2.py            # Figure 2: L2RE vs loss scatter
│   ├── reproduce_fig3_7.py          # Figures 3 & 7: spectral density (full + per-component)
│   ├── reproduce_fig8.py            # Figure 8: min/median/max across widths
│   ├── reproduce_table2_fig145.py   # Table 2, Figures 1, 4, 5: NNCG fine-tuning
│   └── reproduce_table3.py          # Table 3: per-iteration wall-clock times
├── README.md
└── requirements.txt
```

---

## 2. Environment setup

The paper used Python 3.10.12, PyTorch 2.0.0 (CUDA 11.8) on a single NVIDIA Titan V.
Any modern CUDA-capable GPU works; a CPU fallback is possible but slow (especially the
wave PDE under NNCG).

```bash
# (recommended) create a fresh environment
python -m venv .venv && source .venv/bin/activate

# install pinned dependencies (CUDA 11.8 build of torch)
pip install -r requirements.txt

# CPU-only fallback
# pip install torch==2.0.0 torchvision==0.15.0 \
#     --index-url https://download.pytorch.org/whl/cpu
# pip install -r requirements.txt
```

`pyhessian` is **optional**: the repository ships its own pure-PyTorch Stochastic
Lanczos Quadrature estimator in `src/hessian/spectral_density.py`.

---

## 3. Problem definitions (Appendix A)

| PDE        | Equation                          | Domain            | IC / BC                                   | Parameter |
|------------|-----------------------------------|-------------------|-------------------------------------------|-----------|
| Convection | `du/dt + beta du/dx = 0`          | `x∈(0,2π), t∈(0,1)` | `u(x,0)=sin(x)`, periodic BC            | `beta=40` |
| Reaction   | `du/dt - rho u(1-u) = 0`          | `x∈(0,2π), t∈(0,1)` | `u(x,0)=exp(-(x-π)²/(2(π/4)²))`, periodic BC | `rho=5` |
| Wave       | `d²u/dt² - 4 d²u/dx² = 0`         | `x∈(0,1), t∈(0,1)`  | `u(x,0)=sin(πx)+0.5 sin(βπx)`, `u_t(x,0)=0`, `u(0,t)=u(1,t)=0` | `beta=5` |

Data sampling (Section 2.2): 10,000 residual points sampled from the interior of a
255×100 grid, 257 equally spaced IC points, 101 equally spaced BC points per boundary.
The L2 relative error (Eq. 3) is evaluated over the union of the 255×100 grid, the IC
points, and the BC points.

---

## 4. Quick start

### 4.1 Train a single model

```bash
cd opt_for_pinns

# Adam + L-BFGS on convection, width 200, seed 0
python main.py train --pde convection --optimizer adam_lbfgs --width 200 --seed 0

# Plain Adam / plain L-BFGS baselines
python main.py train --pde reaction --optimizer adam --width 100 --seed 0
python main.py train --pde wave --optimizer lbfgs --width 100 --seed 0

# NNCG fine-tuning (runs Adam+L-BFGS first, then 2000 NNCG steps)
python main.py train --pde convection --optimizer nncg --width 200 --seed 0 --mu 1e-2
```

Results are written to `results/<pde>/<optimizer>/w<width>_s<seed>/` as
`result.json` (metrics), `history.json` (loss / grad-norm curves) and
`checkpoint.pt` (model state).

### 4.2 Evaluate a checkpoint

```bash
python main.py eval --pde convection --width 200 --seed 0 \
    --checkpoint results/convection/adam_lbfgs/w200_s0/checkpoint.pt
```

### 4.3 Hessian spectral density

```bash
python main.py spectral --pde convection --width 200 --seed 345 \
    --switch 11000 --num-matvecs 100
```

---

## 5. Reproducing the paper

All reproduction scripts are read-only by default: they consume the artifacts produced
by `run_experiments.py`. Pass `--run` to (re)generate those artifacts first.

### 5.1 Full experiment grid (Table 1, Figures 2 & 8)

```bash
cd opt_for_pinns

# PDE x optimizer x width x seed, with per-(pde, optimizer) learning-rate tuning
python run_experiments.py \
    --pdes convection reaction wave \
    --optimizers adam lbfgs adam_lbfgs \
    --widths 50 100 200 400 \
    --seeds 0 1 2 3 4 \
    --total-iters 41000 --switch 11000
```

This writes `results/summary.json` plus one `result.json` per run. Learning rates are
tuned by selecting the value with the smallest final L2RE (grid
`{1e-5, 1e-4, 1e-3, 1e-2, 1e-1}` for Adam and Adam+L-BFGS; `1.0` for L-BFGS) and cached
in `results/<pde>/best_lr.json`.

Then:

```bash
python scripts/reproduce_table1.py            # Table 1
python scripts/reproduce_fig2.py              # Figure 2 (L2RE vs loss scatter)
python scripts/reproduce_fig8.py              # Figure 8 (min/median/max vs width)
```

**Expected (Table 1, best loss / L2RE):**

| PDE        | Adam              | L-BFGS            | Adam + L-BFGS     |
|------------|-------------------|-------------------|-------------------|
| Convection | 1.40e-4 / 5.96e-2 | 1.51e-5 / 8.26e-3 | **5.95e-6 / 4.19e-3** |
| Reaction   | 4.73e-6 / 2.12e-2 | 8.93e-6 / 3.83e-2 | **3.26e-6 / 1.92e-2** |
| Wave       | 2.03e-2 / 3.49e-1 | 1.84e-2 / 3.35e-1 | **1.12e-3 / 5.52e-2** |

Adam + L-BFGS achieves the lowest loss on every PDE; Figure 2 shows a monotone trend
between final loss and final L2RE.

### 5.2 Hessian spectral density (Figures 3 & 7)

```bash
# Convection -> Figure 3 (full + preconditioned on top, per-component on bottom)
python scripts/reproduce_fig3_7.py --pde convection --width 200 --switch 11000

# Reaction & Wave -> Figure 7 (per-component)
python scripts/reproduce_fig3_7.py --pde reaction --width 200 --switch 11000
python scripts/reproduce_fig3_7.py --pde wave     --width 200 --switch 11000

# or all three at once
python scripts/reproduce_fig3_7.py --all
```

The script trains Adam + L-BFGS to the 11k switch point while recording the L-BFGS
curvature pairs `{s_k, y_k, rho_k}`, then estimates:

* the **full** Hessian spectral density via SLQ,
* the **L-BFGS-preconditioned** density using the unrolled preconditioner
  `H̃_kᵀ H_L H̃_k` (Algorithms 2 & 3),
* the **per-component** densities for the residual, IC and BC loss terms.

**Expected:** large outlier eigenvalues (>1e4 convection, >1e3 reaction, >1e5 wave) with
a large density mass near zero; L-BFGS preconditioning reduces the top eigenvalue by at
least 1e3; the residual component is the most ill-conditioned.

### 5.3 NNCG fine-tuning (Table 2, Figures 1, 4, 5)

```bash
python scripts/reproduce_table2_fig145.py --run \
    --pdes convection reaction wave --width 200 --seed 0 \
    --switch 11000 --total-iters 41000 --finetune-steps 2000
```

For each PDE the script:

1. runs Adam + L-BFGS for 41,000 iterations (switch at 11k),
2. clones the solution and runs **NNCG** for 2,000 steps (tuning `mu ∈ {1e-2, 1e-1}`),
3. clones the solution and runs **gradient descent** for 2,000 steps as a baseline,
4. records loss / L2RE / gradient norm and absolute-error heatmaps at each stage.

**Expected (Table 2, loss / L2RE):**

| PDE        | Adam + L-BFGS     | + NNCG            | + GD        |
|------------|-------------------|-------------------|-------------|
| Convection | 5.95e-6 / 4.19e-3 | **3.63e-6 / 1.94e-3** | unchanged |
| Reaction   | 5.26e-6 / 1.92e-2 | **2.89e-7 / 9.92e-3** | unchanged |
| Wave       | 1.12e-3 / 5.52e-2 | **6.13e-5 / 1.27e-2** | unchanged |

NNCG reduces the loss by more than 10x and improves L2RE, while GD makes no progress
(Figure 4). Figure 5 shows progressive improvement of the absolute-error heatmaps from
Adam → +L-BFGS → +NNCG. Figure 1 shows the wave PDE convergence: Adam is slow,
Adam + L-BFGS stalls around 40,000 steps, and NNCG improves further.

### 5.4 Per-iteration wall-clock (Table 3)

```bash
python scripts/reproduce_table3.py --run \
    --pdes convection reaction wave --width 200 --seed 0
```

Measures the mean seconds per iteration of L-BFGS vs NNCG after reaching the switch
point (with warm-up iterations and CUDA synchronization).

**Expected (seconds/iteration, L-BFGS vs NNCG, ratio):**

| PDE        | L-BFGS | NNCG  | Ratio  |
|------------|--------|-------|--------|
| Convection | 4.6e-2 | 2.5e-1 | 5.43  |
| Reaction   | 3.6e-2 | 7.2e-1 | 20.0  |
| Wave       | 9.0e-2 | 2.9e1  | 322.2 |

Absolute timings are hardware dependent; the qualitative ordering (NNCG slower, ratio
increasing convection → reaction → wave) is the reproduction target.

---

## 6. Algorithms

| Algorithm | Name                              | File                              |
|-----------|-----------------------------------|-----------------------------------|
| 1         | PINN training (Adam + L-BFGS)     | `src/optimizers/adam_lbfgs.py`    |
| 2         | Unrolling the L-BFGS update       | `src/hessian/lbfgs_unroll.py`     |
| 3         | Preconditioned Hessian matvec     | `src/hessian/precond_matvec.py`   |
| 4         | NysNewton-CG                      | `src/optimizers/nncg.py`          |
| 5         | RandomizedNyströmApproximation    | `src/optimizers/nystrom.py`       |
| 6         | NystromPCG                        | `src/optimizers/pcg.py`           |
| 7         | Armijo line search                | `src/optimizers/armijo.py`        |

**NNCG hyperparameters (Algorithm 4):** `eta=1`, `K=2000`, `s=60`, `F=20`,
`mu ∈ {1e-2, 1e-1}` (tuned), `eps=1e-16`, `M=1000`, `alpha=0.1`, `beta=0.5`.
The Nyström approximation is refreshed every `F=20` iterations and PCG is warm-started
with the previous Newton direction `d_{k-1}`. All Hessian-vector products use the
Pearlmutter double-backward trick, so the Hessian is never materialized.

---

## 7. Configuration

Each PDE has a YAML config in `configs/`:

```yaml
name: convection
pde_kwargs: {beta: 40.0}
data: {n_res: 10000, n_ic: 257, n_bc: 101, n_x_grid: 255, n_t_grid: 100}
training: {total_iters: 41000, switch_iters: [1000, 11000, 31000], default_switch: 11000, lbfgs_memory: 100}
lr_grid: {adam: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], lbfgs: [1.0], adam_lbfgs: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
best_lr: {adam: 1e-4, lbfgs: 1.0, adam_lbfgs: 1e-4}
spectral: {width: 200, switch_iter: 11000, lr: 1e-4, seeds: [345, 456, 567], num_matvecs: 100, num_bins: 200}
nncg: {steps: 2000, mu_grid: [1e-2, 1e-1], s: 60, F: 20, M: 1000, eps: 1e-16, eta: 1.0, alpha: 0.1, beta: 0.5}
```

The `best_lr` values are the result of the smallest-L2RE selection procedure; the
spectral seeds `[345, 456, 567]` are the paper's chosen seeds for the spectral-density
figures.

---

## 8. Notes on reproduction

* **Success criteria are qualitative.** The ordering of optimizers, the ill-conditioning
  of the loss, and the NNCG improvement should match the paper; exact numeric values
  depend on hardware, RNG, and library versions and need only agree in order of
  magnitude.
* **L-BFGS termination** uses PyTorch's default gradient/function tolerances, which can
  cause early stopping; the wrapper invokes L-BFGS one iteration at a time so that
  per-iteration logging and curvature recording remain possible.
* **Indefinite Hessians** are handled by the fail-safe eigendecomposition branch of
  Algorithm 5 (`src/optimizers/nystrom.py`).
* **Out of scope** (not implemented): Section 6.2, Section 8, and Figures 6, 9, 10.

---

## 9. Citation

```bibtex
@inproceedings{rathore2024challenges,
  title     = {Challenges in Training PINNs: A Loss Landscape Perspective},
  author    = {Rathore, Pratik and Lei, Weimu and Frangella, Zachary and Lu, Lu and
               Udell, Madeleine},
  booktitle = {International Conference on Machine Learning},
  year      = {2024}
}
```

Reference implementation: <https://github.com/pratikrathore8/opt_for_pinns>
