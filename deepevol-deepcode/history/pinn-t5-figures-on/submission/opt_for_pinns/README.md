# Challenges in Training PINNs: A Loss Landscape Perspective

Code for reproducing the experiments in *"Challenges in Training PINNs: A Loss
Landscape Perspective"*. This repository studies the ill-conditioning of the PINN
loss caused by the differential operators in the residual term, compares Adam,
L-BFGS and Adam+L-BFGS, and introduces **NysNewton-CG (NNCG)** — a matrix-free
second-order optimizer (damped Newton + NyströmPCG + Armijo line search) that
further improves the PINN loss and L2RE after Adam+L-BFGS stalls.

---

## 1. Repository layout

```
opt_for_pinns/
├── main.py                      # Entry point: single training runs + experiment dispatch
├── configs/
│   └── default.yaml             # PDE coefficients, optimizer schedules, hyperparameters
├── src/
│   ├── pdes.py                  # Convection, Reaction, Wave: residual ops, IC/BC, exact solutions
│   ├── model.py                 # MLP (tanh, 3 hidden layers, Xavier init, zero biases)
│   ├── loss.py                  # PINN loss (residual + IC + BC), L2RE metric
│   ├── data.py                  # Grid/point sampling (255x100 interior, 257 IC, 101 BC)
│   ├── train.py                 # Training loops: Adam, L-BFGS, Adam+L-BFGS, +NNCG/+GD
│   ├── optimizers/
│   │   ├── nncg.py              # Algorithm 4: NysNewton-CG
│   │   ├── nystrom.py           # Algorithm 5: RandomizedNystromApproximation
│   │   ├── pcg.py               # Algorithm 6: NystromPCG
│   │   └── armijo.py            # Algorithm 7: Armijo line search
│   ├── hessian.py               # Hessian-vector products (Pearlmutter), SLQ spectral density
│   ├── lbfgs_precond.py         # Algorithms 2 & 3: L-BFGS unrolling + preconditioned matvec
│   └── utils.py                 # Seeding, logging, checkpointing, timing
├── experiments/
│   ├── run_optimizer_comparison.py   # Table 1, Figure 8 (Adam vs L-BFGS vs Adam+L-BFGS)
│   ├── run_spectral_density.py       # Figures 3 & 7 (Hessian / preconditioned Hessian)
│   ├── run_loss_vs_l2re.py           # Figure 2 (loss vs L2RE scatter)
│   ├── run_nncg_finetune.py          # Table 2, Figures 1, 4, 5 (NNCG/GD after Adam+L-BFGS)
│   └── run_wallclock.py              # Table 3 (per-iteration times)
├── requirements.txt
└── README.md
```

---

## 2. Environment setup

The paper used Python 3.10.12, PyTorch 2.0.0 (CUDA 11.8) and a single NVIDIA GPU
(Titan V). A CPU fallback works but is slow.

```bash
# (recommended) create a fresh environment
python3.10 -m venv .venv
source .venv/bin/activate

# install PyTorch 2.0.0 with CUDA 11.8 (see requirements.txt for details)
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118

# remaining dependencies
pip install -r opt_for_pinns/requirements.txt
```

Notes:
- Hessian-vector products use double backprop (`create_graph=True`).
- Training runs in `float32`; Hessian / PCG / SLQ computations use `float64`
  for numerical stability (controlled by `runtime.hessian_dtype` in the config).
- `pyhessian` is optional — a custom SLQ implementation is provided in
  `src/hessian.py`.

---

## 3. Problem definitions (Appendix A)

| PDE        | Operator                     | IC / BC                                              | Domain            |
|------------|------------------------------|------------------------------------------------------|-------------------|
| Convection | `u_t + beta*u_x`, `beta=40`  | `u(x,0)=sin(x)`, periodic `u(0,t)=u(2π,t)`           | `x∈(0,2π), t∈(0,1)` |
| Reaction   | `u_t - rho*u*(1-u)`, `rho=5` | Gaussian IC, periodic BC                             | `x∈(0,2π), t∈(0,1)` |
| Wave       | `u_tt - 4*u_xx`, `beta=5`    | `u(x,0)=sin(πx)+0.5 sin(βπx)`, `u_t(x,0)=0`, Dirichlet | `x∈(0,1), t∈(0,1)` |

Exact solutions are implemented in `src/pdes.py` and used to compute the L2
relative error (L2RE):

```
L2RE = sqrt( Σ(y_i - y'_i)^2 / Σ y'_i^2 )
```

evaluated on the full `255x100` interior grid plus `257` IC points and `101` BC
points per boundary.

---

## 4. Quick start

### 4.1 Single training run

```bash
# Adam + L-BFGS on convection, width 200, seed 345
python opt_for_pinns/main.py train --pde convection --optimizer adam_lbfgs \
    --width 200 --seed 345 --adam-lr 1e-3 --switch-iter 11000 --total-iters 41000

# NNCG fine-tuning after Adam+L-BFGS (2000 extra steps)
python opt_for_pinns/main.py train --pde convection --optimizer nncg \
    --width 200 --seed 345 --mu 1e-2
```

### 4.2 Full experiments

```bash
# Table 1 / Figure 8 — optimizer comparison
python opt_for_pinns/main.py experiment optimizer_comparison --outdir results/optimizer_comparison

# Figure 2 — loss vs L2RE scatter
python opt_for_pinns/main.py experiment loss_vs_l2re --outdir results/loss_vs_l2re

# Figures 3 & 7 — Hessian / preconditioned spectral density
python opt_for_pinns/main.py experiment spectral_density --outdir results/spectral_density

# Table 2 / Figures 1, 4, 5 — NNCG & GD fine-tuning
python opt_for_pinns/main.py experiment nncg_finetune --outdir results/nncg_finetune

# Table 3 — per-iteration wall-clock
python opt_for_pinns/main.py experiment wallclock --outdir results/wallclock

# run everything
python opt_for_pinns/main.py experiment all --outdir results
```

Each experiment script can also be run directly, e.g.:

```bash
python -m opt_for_pinns.experiments.run_optimizer_comparison --outdir results/optimizer_comparison
```

---

## 5. Experiment protocol

- **Seeds**: 5 random seeds per configuration (`123, 234, 345, 456, 567`);
  results are reported as min / median / max.
- **Iterations**: `41000` total for Adam+L-BFGS; `+2000` for NNCG / GD
  fine-tuning.
- **Adam**: learning-rate grid `{1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`, betas
  `(0.9, 0.999)`, eps `1e-8`.
- **L-BFGS**: `lr=1.0`, `history_size=100`, strong-Wolfe line search.
- **Adam+L-BFGS**: Adam then L-BFGS, switching at `{1000, 11000, 31000}`.
- **NNCG hyperparameters**: `η=1`, `K=2000`, `s=60`, `F=20`, `ε=1e-16`,
  `M=1000`, `α=0.1`, `β=0.5`; `μ` tuned in `{1e-5..1e-1}` (best `1e-2`/`1e-1`).
- **Spectral-density figures (3 & 7)** use only the `11000`-iteration switch
  runs, with the per-PDE config (width, Adam lr, seed) that yields the smallest
  L2RE: width `200`; lr `1e-4` (convection), `1e-3` (reaction), `1e-3` (wave);
  seeds `345` / `456` / `567`.
- **Full-batch training**: all residual + IC + BC points are used every step.

---

## 6. Expected outputs

### Table 1 / Figure 8 — optimizer comparison (best loss, L2RE)

| PDE        | Adam              | L-BFGS            | Adam+L-BFGS       |
|------------|-------------------|-------------------|-------------------|
| Convection | 1.40e-4, 5.96e-2  | 1.51e-5, 8.26e-3  | **5.95e-6, 4.19e-3** |
| Reaction   | 4.73e-6, 2.12e-2  | 8.93e-6, 3.83e-2  | **3.26e-6, 1.92e-2** |
| Wave       | 2.03e-2, 3.49e-1  | 1.84e-2, 3.35e-1  | **1.12e-3, 5.52e-2** |

Adam+L-BFGS attains the smallest min loss / L2RE for all PDEs (reaction is the
closest case).

### Figure 2 — loss vs L2RE

Lower loss generally implies lower L2RE; the scatter shows a monotone trend
across PDEs.

### Figures 3 & 7 — Hessian spectral density

- Large outlier eigenvalues (`>1e4` convection, `>1e3` reaction, `>1e5` wave)
  with the bulk of the density near 0.
- L-BFGS preconditioning (dashed) reduces the top eigenvalue by `≥1e3`.
- The residual component is the most ill-conditioned.

### Table 2 / Figures 1, 4, 5 — NNCG & GD fine-tuning

| PDE        | Adam+L-BFGS       | +NNCG             | +GD        |
|------------|-------------------|-------------------|------------|
| Convection | 5.95e-6, 4.19e-3  | **3.63e-7, 1.94e-3** | unchanged |
| Reaction   | 5.26e-6, 1.92e-2  | **2.89e-7, 9.92e-3** | unchanged |
| Wave       | 1.12e-3, 5.52e-2  | **6.13e-5, 1.27e-2** | unchanged |

NNCG reduces the loss by `>10x` on all PDEs and significantly lowers the
gradient norm on convection & wave; GD makes no progress. Figure 5 shows the
pointwise absolute-error heatmaps (3x3 grid: PDEs x {Adam, Adam+L-BFGS,
Adam+L-BFGS+NNCG}) decreasing left→right.

### Table 3 — per-iteration wall-clock

| PDE        | L-BFGS (s) | NNCG (s) | Ratio   |
|------------|------------|----------|---------|
| Convection | 4.6e-2     | 2.5e-1   | 5.43    |
| Reaction   | 3.6e-2     | 7.2e-1   | 20.00   |
| Wave       | 9.0e-2     | 2.9e1    | 322.22  |

NNCG is much slower, especially on the wave equation (second-order derivatives
in the HVP).

---

## 7. Outputs

Every experiment writes to its `--outdir`:

- `results.json` — raw records plus min/median/max summaries.
- Figures as PNG files, e.g. `figure8_optimizer_comparison.png`,
  `figure2_loss_vs_l2re.png`, `figure3_spectral_density.png`,
  `figure7_component_density.png`, `figure1_loss_curves.png`,
  `figure4_gradnorm_curves.png`, `figure5_error_heatmaps.png`.

---

## 8. Out of scope

The following are **not** reproduced here: Section 6.2, Section 8, Figures 6, 9,
10, and Appendices F, G.

---

## 9. Reference

Original reference implementation: <https://github.com/pratikrathore8/opt_for_pinns>
