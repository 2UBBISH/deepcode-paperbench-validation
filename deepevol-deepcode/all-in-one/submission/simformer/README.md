# Simformer — Reproduction

Simulation-based inference with probabilistic diffusion models.

This repository is a complete, runnable reproduction of the paper

> **Simformer: Simulation-based inference with probabilistic diffusion models** — Olsson, Wenk, Ludwig, Ma, Sieber, Bengio, Durante, Charpentier, Riquelme, Oh, Engelcke, Macke (2024).

It implements a transformer-parameterised score-based diffusion model trained on the **joint**
`p(θ, x)` with *structure-aware attention masks* (`M_E`) and *per-sample condition masks* (`M_C`).
A single amortized network therefore samples **all** conditionals:

* posterior `p(θ | x)`
* likelihood `p(x | θ)`
* parameter conditionals `p(θ_i | θ_j, x)`
* arbitrary conditional targets (Sec. 4.1 protocol)
* joint `p(θ, x)`

and it natively supports missing / unstructured data, function-valued parameters, and
interval/constraint conditioning via diffusion guidance (Algorithm 1, Sec. 3.4).

---

## 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.10+ is expected. See `requirements.txt` for the pinned stack
(`numpy`, `scipy`, `torch`, `jax`/`optax`/`hydra-core`, `sbi`, `scikit-learn`, `matplotlib`,
`emcee`, `pytest`).

> **Backend note (deliberate design decision).** The paper's mathematics (VESDE/VPSDE
> coefficients, Anderson reverse SDE, Euler–Maruyama, probability-flow ODE, Algorithm 1
> guidance, MCMC references, mask construction) lives in a *framework-agnostic NumPy core*,
> while the neural score network (tokenizer + transformer + optional CNN embedding net) is
> implemented in **PyTorch**. The official JAX + hydra stack is retained in `requirements.txt`
> for config parity and reference runs. Default hyper-parameters follow Appendix A2.1 exactly.

Hardware: a GPU is recommended (transformer + 500-step reverse SDE over many samples). CPU-only
works for the smallest benchmark budgets (1k simulations, reduced steps).

---

## 2. Repository layout

```
simformer/
  simformer/                     # core library
    tokenizer.py                 # SBI tokenizer: id/value/condition/metadata embeddings  (§3.1)
    transformer.py               # transformer score network + time Fourier injection      (§2.2/§3)
    diffusion.py                 # VESDE/VPSDE, reverse SDE, probability-flow ODE          (§2.3/A2.1)
    condition_masks.py           # M_C sampling (joint/posterior/likelihood/rand)          (§3.1/§3.3)
    attention_masks.py           # directed/undirected task masks, symmetrisation         (§3.2/A1.1)
    graph_inversion.py           # Webb (2018) graph inversion -> dynamic masks            (§3.2/A1.1)
    training.py                  # masked denoising score matching                         (§3.3)
    sampling.py                  # arbitrary-conditional reverse-SDE sampling              (§3.3/A1.3)
    guidance.py                  # Algorithm 1 general guidance + constraints              (§3.4/A3.3)
    embedding_nets.py            # optional CNN embedding net for high-dim data            (§3.1/A3.2)
    utils.py                     # score math, RFF, densities, mask helpers
  tasks/                         # simulators (Sec. 4.1-4.4, Appendix A2.2)
    gaussian_linear.py  gaussian_mixture.py  two_moons.py  slcp.py
    tree.py  hmm.py  lotka_volterra.py  sird.py  hodgkin_huxley.py
    gravitational_waves.py
  baselines/
    npe_nle_nre.py               # sbi wrappers (NPE/NLE/NRE, NSF)                          (A2.1/A3.1)
    npse.py                      # conditional-MLP score baseline + posterior-only Simformer
  reference/
    mcmc.py                      # slice+MH and HMC ground-truth conditional samplers       (A2.2)
  eval/
    c2st.py                      # random-forest C2ST (100 trees)                          (§4.1)
    coverage.py                  # expected-coverage calibration                           (A3.1)
    nll.py                       # probability-flow-ODE NLL                                (A3.1)
  configs/
    task/*.yaml  model/*.yaml
  scripts/
    train.py                     # train Simformer on a task
    sample.py                    # sample conditionals / guided sampling
    run_experiments.py           # full benchmark sweep (Fig. 4, ablation, scientific tasks)
  tests/                         # tokenizer / attention masks / graph inversion / diffusion / guidance
  requirements.txt
```

---

## 3. Quick start

### 3.1 Train

```bash
python -m simformer.scripts.train \
    --task two_moons --budget 10000 --mask directed \
    --sde vesde --max-steps 50000 --outdir runs/two_moons_directed
```

Key flags:

| flag | meaning |
| --- | --- |
| `--task` | task name (`two_moons`, `slcp`, `gaussian_linear`, `gaussian_mixture`, `tree`, `hmm`, `lotka_volterra`, `sird`, `hodgkin_huxley`) |
| `--budget` | number of joint simulations used for training (`1_000` / `10_000` / `100_000`) |
| `--mask` | attention-mask variant: `dense`, `undirected`, `directed` (directed uses graph inversion per batch) |
| `--sde` | `vesde` (default, σ_max=15, σ_min=1e-4) or `vpsde` (β_min=0.01, β_max=10) |
| `--max-steps`, `--batch-size`, `--lr` | optimisation schedule (Adam, batch 1000, lr 3e-4, warmup+cosine, grad-clip 1.0) |
| `--dry-run` | tiny smoke run |
| `--verify-sampling` | runs a short reverse-SDE sanity check after training |

Outputs: `model.pt`, `history.json`, `dataset.npy`, `train_meta.json` and (optionally)
`sampling_check.json`.

### 3.2 Sample arbitrary conditionals

```bash
# posterior p(theta | x)
python -m simformer.scripts.sample --ckpt runs/two_moons_directed/model.pt \
    --task two_moons --mode posterior --n-samples 1000 --n-steps 500

# 100 random conditional targets (Sec. 4.1 protocol)
python -m simformer.scripts.sample --ckpt runs/two_moons_directed/model.pt \
    --task two_moons --mode arbitrary --n-targets 100 --c2st

# interval-guided sampling (Algorithm 1)
python -m simformer.scripts.sample --ckpt runs/hh_directed/model.pt \
    --task hodgkin_huxley --mode guided --lower 0 0 --upper 1 1 \
    --indices 2 3 --self-recurrence 5
```

### 3.3 Full benchmark sweep

```bash
# benchmark C2ST + baselines + arbitrary conditionals + reverse-SDE ablation + scientific tasks
python -m simformer.scripts.run_experiments --experiments benchmark baselines arbitrary reverse_sde scientific \
    --tasks gaussian_linear gaussian_mixture two_moons slcp \
    --budgets 1000 10000 100000 \
    --variants dense undirected directed \
    --baselines npe nle nre npse \
    --outdir results

# quick smoke run of everything
python -m simformer.scripts.run_experiments --quick --outdir results_quick
```

Results are written to `results/results.json` plus a Fig. 4-style PNG.

---

## 4. What is implemented (paper mapping)

### Core (§2–3)

* **Tokenizer (§3.1, Addendum "Tokenization")** — one token per variable of `(θ, x)`, built by
  concatenating `[identifier embedding | value embedding | metadata | condition-state embedding]`.
  Value = scalar repeated to `token_dim = 50`. Condition state `True → learnable embedding`,
  `False → zeros`. Function-valued parameters share one identifier embedding plus a random
  Fourier embedding of the index set.
* **Transformer score network (§2.2, §3, §A2.1)** — 6 layers (8 for Lotka-Volterra / SIRD /
  Hodgkin-Huxley), 4 heads, attention size 10, widening factor 3 (FF hidden 150), token dim 50;
  diffusion time enters as a 128-dim random Gaussian Fourier embedding added to every FF block
  output; scaled dot-product attention with mask `M_E`.
* **Diffusion / SDE (§2.3, §A2.1, §A3.1)** — VESDE (`f = 0`,
  `g(t) = σ_min (σ_max/σ_min)^t √(2 log(σ_max/σ_min))`) and VPSDE
  (`f = −½ β(t) x`, `g = √β(t)`), `σ_max = 15`, `σ_min = 1e-4`, `β_min = 0.01`, `β_max = 10`,
  `t ∈ [1e-5, 1]`. Anderson reverse SDE solved by Euler–Maruyama, default 500 steps (≥ 50
  sufficient, see ablation). Probability-flow ODE for log-likelihood.
* **Condition-mask sampler (§3.1/§3.3, §A2.1)** — per batch element `M_C` is drawn uniformly from
  {joint (all False), posterior (params False/data True), likelihood (data False/params True),
  Bernoulli(0.3), Bernoulli(0.7)}; partially noised input
  `x̂_t^{M_C} = (1 − M_C)·x̂_t + M_C·x̂_0`.
* **Attention-mask builder (§3.2, §A1.1)** — task-specific directed base masks (Gaussian Linear,
  Two Moons/Gaussian Mixture, SLCP, Tree, HMM; Lotka-Volterra metadata-dependent) plus undirected
  symmetrisation; `mask_variants()` returns the `dense`/`undirected`/`directed` trio of Fig. 4.
* **Graph inversion (§3.2, §A1.1)** — Webb (2018) algorithm: moralise `G`, min-fill ordering over
  latent variables, add edges to `H`; final mask = `M_E + edges(H)`. Needed for directed masks
  under conditioning.
* **Training loop (§3.3, §A2.1)** — masked denoising score matching
  `ℓ = (1 − M_C)·(s_φ^{M_E}(x̂_t^{M_C}, t) − ∇ log p_t(x̂_t|x̂_0))`, `L = E[‖ℓ‖²]`;
  batch 1000, Adam, random `t` per sample, early stopping on validation loss. Score target
  `∇ log p_t = −ε/σ_t`.
* **Sampling (§3.3, §A1.3)** — draw terminal noise, run the reverse SDE only on unobserved
  variables while clamping observed ones; score combination for i.i.d. data and post-hoc
  prior/likelihood adaptation.
* **Guidance (§3.4, §A1.3, §A3.3 Algorithm 1)** — self-recurrence `r` steps; constraint score on
  the denoised estimate `x̂_0 = (x̂_t + σ² s)/μ`; score modified by `∇ log sigmoid(−s(t)c(x̂))`
  with `s(t) = 1/σ(t)²`; interval constraint `c = x̂ − u`.

### Tasks (§A2.2, §4.2–4.4)

`gaussian_linear`, `gaussian_mixture`, `two_moons`, `slcp` (benchmark), plus `tree`, `hmm`
(arbitrary-conditional references), plus the scientific simulators:

* **Lotka-Volterra** — ODEs with Gaussian noise (σ = 0.1), sigmoid-transformed Normal prior on
  `θ ∈ (1, 3)`, posterior on a time grid `t ∈ [0, 15]`.
* **SIRD** — ODEs with GP-RBF contact rate `β(t) = sigmoid(GP(0, k))`, log-normal observation
  noise σ = 0.05, function-valued `β` on a 20-point index grid.
* **Hodgkin-Huxley** — Pospischil-style HH dynamics with the stated rate functions / `efun` and
  sodium-charge → energy conversion; 7 voltage summary statistics (spike count, resting
  mean/std, spiking mean, 2nd/3rd/4th central moments) + metabolic energy in µJ/s;
  interval-guided energy conditioning (§4.4).
* **Gravitational waves** (optional) — reduced-order inspiral simulator producing two 8192-dim
  detector measurements, compressed by a CNN embedding net into one token each; targeted
  conditionals `p(θ|x₁,x₂)`, `p(θ|x₁)`, `p(θ|x₂)`.

### Baselines & evaluation (§4.1, §A2.1, §A3.1)

* `npe`, `nle`, `nre` via `sbi` (neural spline flow, defaults), with self-contained PyTorch
  fallbacks (conditional MAF / ratio classifier + MCMC).
* `npse` — conditional-MLP score network; `npse_simformer` — posterior-only Simformer variant.
* **C2ST** — random forest with 100 trees (0.5 = perfect, 1.0 = fully separable).
* **Coverage** — expected coverage on HDR regions (k-NN / Mahalanobis / exact-log-density /
  per-dimension interval estimators).
* **NLL** — probability-flow-ODE log-likelihood on 5000 joint samples.
* **Reference MCMC** — Two Moons/SLCP slice + MH; Tree/HMM 5000-step HMC; keep the last sample
  per chain (Appendix A2.2).

---

## 5. Tests

```bash
pytest simformer/tests -q
```

* `test_tokenizer.py` — True → embedding / False → zeros, scalar repetition, function-valued
  tokens, subsampling, layout invariants.
* `test_attention_masks.py` — masks match the paper's block matrices, registry dispatch,
  dense/undirected/directed variants.
* `test_graph_inversion.py` — moralisation, min-fill ordering, posterior/likelihood conditioning
  edge insertion, batched equivalence, no input mutation.
* `test_diffusion.py` — VESDE/VPSDE coefficients and marginals, score↔ε algebra, Tweedie
  denoising, reverse-SDE step, probability-flow ODE, divergence estimators.
* `test_guidance.py` — constraint objects, scaling functions, guided score, Algorithm 1
  self-recurrence, interval enforcement, conditioned-dimension clamping.

---

## 6. Reproducing paper results

| Paper result | How to reproduce |
| --- | --- |
| **Fig. 4** — C2ST vs simulation budget (1k/10k/100k) for Gaussian Linear, Gaussian Mixture, Two Moons, SLCP, comparing Simformer (dense/undirected/directed) and NPE | `python -m simformer.scripts.run_experiments --experiments benchmark baselines` |
| **Arbitrary conditionals** — C2ST of Simformer conditionals vs MCMC references on 4 benchmark tasks + Tree + HMM (100 random targets) | `--experiments arbitrary` |
| **Fig. A7** — reverse-SDE evaluation-step ablation (near-best above ~50 steps) | `--experiments reverse_sde` |
| **Sec. 4.2 / Fig. 5** — Lotka-Volterra posterior covers true parameters; posterior predictive realistic; adding predator observations reduces uncertainty | `python -m simformer.scripts.train --task lotka_volterra` then `sample --mode posterior` |
| **Sec. 4.3 / Fig. 6** — SIRD recovers `β(t)` and constant rates; uncertainty rises where infections ≈ 0 | `--task sird` + `tasks.sird.figure6_scenarios()` |
| **Sec. 4.4 / Fig. 7** — HH energy-interval guidance constrains the posterior (`g_Na`, `g_K`) while voltage predictions stay consistent | `--task hodgkin_huxley` + `tasks.hodgkin_huxley.energy_constraint` + `sample --mode guided --self-recurrence 5` |

Expected outcome: Simformer matches or beats NPE at a given budget (except Gaussian Linear at
10k), and the structured masks help the sparse tasks (Gaussian Linear, SLCP). C2ST → 0.5 means
indistinguishable from the ground-truth posterior.

**Declared as not required for replication:** the calibration/NLL analyses of §4.1 and §4.3 and
the detailed guidance study of §4.4 (their implementations are provided, `eval/coverage.py`,
`eval/nll.py`, `guidance.py`, but they are not part of the required success criterion).

---

## 7. Defaults chosen where the paper is silent

* Score parameterisation is ε-prediction: `∇ log p = −ε/σ_t`, loss target `−ε/σ_t`.
* Loss weighting `λ(t) = 1`.
* Adam `lr ≈ 3e-4` with linear warmup + cosine decay, gradient-norm clip 1.0.
* "Attention size 10" = per-head dimension (4 heads × 10 = 40, projected back to 50).
* No explicit positional encoding.
* SIRD's second global prior is treated as the mean `μ` (the printed "delta" is a typo).
* Two Moons `r ≈ 0.1`, `r_std = 0.01` (the paper's `0.012` is exposed as `PAPER_R_STD`).

Any deviation from these defaults is documented in the corresponding module docstring.

---

## 8. Citation

```bibtex
@article{olsson2024simformer,
  title   = {Simformer: Simulation-based inference with probabilistic diffusion models},
  author  = {Olsson, Carl and Wenk, Philippe and Ludwig, Jakob and Ma, Chao and Sieber, Jan
             and Bengio, Yoshua and Durante, Danilo and Charpentier, Bertrand and Riquelme, Carlos
             and Oh, Sanghyeon and Engelcke, Martin and Macke, Jakob H.},
  year    = {2024}
}
```

Official code: <https://github.com/mackelab/simformer>
