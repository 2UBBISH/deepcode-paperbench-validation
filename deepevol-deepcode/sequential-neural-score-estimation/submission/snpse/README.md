# SNPSE — Sequential Neural Posterior Score Estimation

Reproduction code for **Sequential Neural Posterior Score Estimation** (SNPSE).

The repository implements:

* **NPSE** — a conditional score-based diffusion model that learns the score of the
  perturbed posterior `∇_θ log p_t(θ_t | x)` for simulator-based inference
  (Section 2.2, objective eq. 7).
* **TSNPSE** — the sequential variant (Section 3.1, Algorithm 1). Each round draws
  parameters from the *HPR-truncated prior* `p̄^{r-1}` (highest-probability region of the
  previous posterior approximation), simulates data, appends it to the accumulated
  dataset `D`, and retrains the score network on the TSNPSE objective (eq. 11).
  By construction (`Proposition 3.1`) **no importance-weight correction is required**.
* **SNPSE-A / SNPSE-B / SNPSE-C** — the alternative sequential schemes of Section 3.2 /
  Appendices C.2–C.4 (post-hoc SIR correction, importance-weighted DSM loss,
  score-space correction) used in the ablations.
* **NLSE** — Neural Likelihood Score Estimation (Appendix B), which learns the perturbed
  **likelihood** score (eq. 57) and adds a perturbed **prior** score — analytic (eq. 58/59)
  or learned with Algorithm 2 / eq. 64.

Everything is implemented in plain PyTorch. `sbibm` (benchmark tasks, NPE/SNPE-C
baselines, C2ST) and the mackelab `tsnpe_neurips` package (pyloric simulator, TSNPE
baseline) are **optional**: every consumer module ships a dependency-free fallback.

---

## 1. Repository layout

```
snpse/
├── main.py                          # CLI entry: --method {npse,tsnpse,snpse-a,-b,-c,nlse}
├── snpse/
│   ├── __init__.py                  # public API facade
│   ├── score_network.py             # s_psi: θ/x/t embeddings + head MLP (+ energy variant)
│   ├── sdes.py                      # VE SDE, VP SDE, transitions, analytic score targets
│   ├── losses.py                    # DSM losses (eq 7, 11, 15, 79, 99, 102, 57, 64)
│   ├── sampler.py                   # prob-flow ODE (eq 4) + reverse SDE (eq 3) + CoV (eq 5)
│   ├── trainer.py                   # shared Adam loop, 15% early stopping
│   ├── npse.py                      # NPSE (round-1 / amortised) train + sample
│   ├── tsnpse.py                    # TSNPSE sequential driver (Algorithm 1)
│   ├── hpr.py                       # HPR_eps + truncated-proposal rejection sampler
│   ├── snpse_variants.py            # SNPSE-A/B/C (Algorithms 3, 4, 5)
│   ├── nlse.py                      # NLSE + prior-score estimation (Algorithm 2)
│   └── utils.py                     # standardisation, batch sampling, seeding
├── tasks/
│   ├── benchmarks.py                # 8 sbibm benchmark tasks (+ fallbacks)
│   ├── pyloric.py                   # Pyloric simulator wrapper (mackelab)
│   └── c2st.py                      # C2ST metric via sbibm
├── experiments/
│   ├── run_benchmarks.py            # NPSE/TSNPSE × 8 tasks × 3 budgets × {VE,VP}
│   ├── run_pyloric.py               # neuroscience experiment (Figures 4, 7, 8)
│   ├── run_ablations.py             # SNPSE-A/B/C, NLSE, VE-vs-VP
│   └── baselines.py                 # NPE/SNPE-C (sbibm), TSNPE (mackelab)
├── configs/
│   ├── default.yaml                 # all hyperparameters / experiment grids
│   └── sde_config.yaml              # VE/VP settings, sampler & HPR defaults
├── requirements.txt
└── README.md
```

---

## 2. Environment setup

Python ≥ 3.9 with PyTorch.

```bash
cd snpse
pip install -r requirements.txt
```

`requirements.txt` installs `torch`, `numpy`, `scipy`, `sbibm`, `pyyaml`, `tqdm`,
`pandas`, `matplotlib` and the optional `torchdiffeq` / `scikit-learn`. The ODE solver
used by default is a self-contained Dormand–Prince 5(4) RK45 implementation in
`snpse/sampler.py`, so `torchdiffeq` is not needed.

Two non-pip resources (documented in `requirements.txt`):

```bash
# Pyloric simulator (31 parameters → 18 summary statistics) and the TSNPE baseline
git clone https://github.com/mackelab/tsnpe_neurips.git
export PYTHONPATH=$PYTHONPATH:$(pwd)/tsnpe_neurips

# Reference implementation of the paper
git clone https://github.com/jacksimons15327/snpse_icml.git
```

If `sbibm` / the mackelab repo are absent, the code falls back to:
* built-in priors/simulators/reference observations for the eight benchmark tasks,
* built-in mixture-density-network NPE and APT SNPE-C baselines,
* a deterministic torch pyloric fallback (`PyloricSimulator(backend="fallback")`),
* a scikit-learn / pure-torch C2ST classifier.

Hardware: benchmarks are small MLPs (a single GPU is plenty, CPU works); the pyloric
runs are simulator-bound rather than GPU-bound.

---

## 3. Quick start

```bash
cd snpse

# Fast end-to-end self-check (imports, NPSE on a Gaussian-linear task, C2ST, TSNPSE round-1)
python main.py --smoke-test

# Unit-level diagnostics
python -m snpse.sdes            # closed-form VE/VP score targets (finite-difference check)
python -m snpse.sampler ve      # RK45 ODE + change-of-variables vs analytic Gaussian
python -m snpse.npse            # Gaussian-linear score/posterior recovery
python -m snpse.hpr             # HPR quantile / rejection / mixture normalisation
python -m snpse.nlse            # analytic perturbed prior + eq. 57 + Algorithm 2
python -m tasks.c2st            # C2ST sanity (identical → 0.5, shifted → 1.0)
```

Single inference cell:

```bash
python main.py --method npse    --task two_moons --budget 10000 --sde ve
python main.py --method tsnpse  --task slcp      --budget 10000 --sde vp --num-rounds 10
python main.py --method snpse-a --task gaussian_linear_uniform --budget 10000
python main.py --method snpse-b --task slcp      --budget 10000
python main.py --method snpse-c --task slcp      --budget 10000
python main.py --method nlse    --task two_moons --budget 10000
python main.py --method npe     --task slcp      --budget 10000
```

Results are printed as JSON; pass `--output results/run.json` to persist them.

---

## 4. Experiments

### 4.1 Benchmarks (Section 5.2, Figures 2–3)

Eight sbibm tasks × budgets `{1e3, 1e4, 1e5}` × SDEs `{VE, VP}`:

```bash
python -m experiments.run_benchmarks \
    --methods npse tsnpse --sdes ve vp --budgets 1000 10000 100000 \
    --num-rounds 10 --num-samples 10000 \
    --output results/benchmarks.json --figure results/benchmarks.png
```

Metric: **C2ST** (lower is better, `0.5` = ideal) via `tasks/c2st.py` (sbibm defaults:
10-fold cross-validation, accuracy scoring, MLP classifier with (50, 50) hidden layers).

Baselines (NPE, SNPE-C via `sbibm` with default hyper-parameters; TSNPE via the mackelab
reference implementation — *not* re-implemented here):

```bash
python -m experiments.run_benchmarks --methods npe snpe_c tsnpe \
    --budgets 1000 10000 100000 --output results/baselines.json
```

FMPE numbers are literature values from Dax et al. (2023), see
`experiments/baselines.py:FMPE_PUBLISHED_C2ST`.

### 4.2 Ablations (Section 5.4 / Appendices C, D)

```bash
python -m experiments.run_ablations --output results/ablations.json --figure results/ablations.png
python -m experiments.run_ablations --quick     # smoke-test path
```

Three sections:

1. `variants` — SNPSE-A/B/C vs TSNPSE (and NPSE) on `slcp` and `gaussian_linear_uniform`.
2. `nlse` — NPSE vs NLSE on `two_moons` (Figure 5).
3. `sde` — VE vs VP over all eight tasks, to reproduce the recommended dimensionality
   split (VE preferred for low-dimensional, VP for high-dimensional problems).

`run_ablations.check_expectations` reports whether the observed numbers match the paper's
qualitative claims (TSNPSE dominance, SNPSE-C failure with C2ST → 1, NLSE ≈ NPSE when the
perturbed prior is analytic, SDE preference).

### 4.3 Neuroscience — Pyloric (Section 5.3, Figures 4, 7, 8)

TSNPSE with the **VP** SDE, **9 rounds**, `30000` initial simulations + `20000` per round:

```bash
python -m experiments.run_pyloric --num-rounds 9 \
    --initial-simulations 30000 --simulations-per-round 20000 \
    --num-samples 10000 --output results/pyloric.json --figure results/pyloric.png
```

The driver reports:
* per-round fraction of valid summary statistics (target: ≈ **81 %** in the final round),
* the posterior predictive check against the observed data,
* posterior marginals over the 31 parameters (cf. Deistler et al. 2022a; Gloeckler et al. 2022),
* SBCC coverage, which should be close to the diagonal at high confidence levels.

Invalid summary statistics produced by the simulator are replaced by the
"mean − 2·std" of the prior predictive per Appendix E.2 / Deistler et al. (2022a).

> The TSNPE (mackelab) and SNVI baselines for the pyloric task are **not** replicated here;
> their values are read from the original papers.

---

## 5. Algorithms and where they live

| Method  | Objective / scheme                                   | Section        | Implementation                     |
|---------|------------------------------------------------------|----------------|------------------------------------|
| NPSE    | `E ‖λ_t (s_ψ(θ_t,x,t) − ∇log p_{t\|0})‖²`, eq. 7     | §2.2           | `snpse/npse.py`, `snpse/losses.py` |
| TSNPSE  | eq. 11, θ₀ ~ p̄^r, accumulated data                  | §3.1 Alg. 1    | `snpse/tsnpse.py`                  |
| HPR     | κ = ε-quantile, `ε = 5e-4`; rejection sampler        | §3.1 eq. 9–10  | `snpse/hpr.py`                     |
| SNPSE-A | eq. 79 loss + post-hoc SIR weights eq. 12–14         | §3.2 / §C.2    | `snpse/snpse_variants.py`          |
| SNPSE-B | eq. 15 / eq. 99 importance-weighted loss             | §3.2 / §C.3    | `snpse/snpse_variants.py`          |
| SNPSE-C | eq. 19 / 102 / 103 score-space correction            | §3.2 / §C.4    | `snpse/snpse_variants.py`          |
| NLSE    | eq. 55 (likelihood score eq. 57 + prior score)       | §B.1–§B.2      | `snpse/nlse.py`                    |
| Priors  | analytic perturbed prior, eq. 58 (uniform) / 59 (GMM); else Alg. 2 eq. 64 | §B.2 | `snpse/nlse.py` |

Primitives: forward SDEs and analytic score targets in `snpse/sdes.py`; the score network
`θ/x/t` embedding + head MLP in `snpse/score_network.py`; the probability-flow ODE with the
instantaneous change-of-variables in `snpse/sampler.py`; the shared Adam loop in
`snpse/trainer.py`.

---

## 6. Defaults (paper-silent choices)

The paper leaves several hyper-parameters unspecified. The defaults below are recorded in
`configs/default.yaml` / `configs/sde_config.yaml` and marked `# DEFAULT`:

| Item | Value | Rationale |
|------|-------|-----------|
| DSM weighting `λ_t` | `g(t)²` (SDE-specific) | Song et al. (2021) |
| Time grid | 1000 uniform steps, `t ∈ [0, 1]` | simple, stable |
| Optimiser | `torch.optim.Adam` lr `1e-4`, ≤ 3000 iters | paper §5.1 |
| Validation | 15 % hold-out, patience 1000, restore best | paper §5.1 / §E.3.2 |
| Batch size | 50 / 200 / 500 for 1e3 / 1e4 / 1e5 | paper §E.3.2 |
| ODE solver | RK45 (Dormand–Prince), `atol = rtol = 1e-5` | plan default |
| `σ_min` (VE) | 0.01 if `dim == 2`, else 0.05 | paper §E.3.1 |
| `σ_max` (VE) | Song & Ermon Technique 1 | paper §E.3.1 |
| `β_min, β_max` (VP) | 0.1, 11.0 (linear schedule) | paper §E.3.1 |
| HPR `ε`, samples | `5e-4`, 20000 draws | paper §E.3.3 |
| Weight init | Xavier-uniform, zero bias | unspecified |
| Activations | SiLU everywhere | paper §E.3.2 |

**Addendum rule:** for VE-SDE sequential methods (TSNPSE and SNPSE-A/B/C), `σ_max` is
computed from **first-round data only**.

---

## 7. Expected results

* **NPSE** is comparable to or better than NPE at all budgets.
* **TSNPSE** matches or beats SNPE-C and TSNPE overall, with the clearest wins on
  **SLCP** and **Lotka Volterra** (Figures 2–3).
* **VE** is preferred for low-dimensional tasks, **VP** for high-dimensional ones.
* **SNPSE-A/B/C**: TSNPSE clearly better; SNPSE-C fails (C2ST ≈ 1).
* **NPSE vs NLSE** (Figure 5): similar when the perturbed prior is analytic; NPSE better
  when the prior score must be learned.
* **Pyloric** (Section 5.3): ≈ 81 % valid summary statistics in the final round, posterior
  predictive closely matching the observed data, marginals consistent with the literature,
  and SBCC empirical coverage ≈ nominal at high confidence levels.

---

## 8. Reference

* Paper: *Sequential Neural Posterior Score Estimation*.
* Reference implementation: <https://github.com/jacksimons15327/snpse_icml>
* TSNPE + pyloric simulator: <https://github.com/mackelab/tsnpe_neurips>
* Benchmark tasks, NPE / SNPE-C baselines and C2ST: <https://github.com/sbi-benchmark/sbibm>
