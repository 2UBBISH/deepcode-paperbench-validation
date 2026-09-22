# LBCS — Refined Coreset Selection (Reproduction)

Code reproduction of **Refined Coreset Selection (RCS)** solved by
**Lexicographic Bilevel Coreset Selection (LBCS)**.

RCS is defined as: *minimize the coreset size subject to a model-performance
constraint*, with a **lexicographic priority** `performance > size`. LBCS solves
RCS with a black-box randomized direct-search outer loop (**LexiFlow**, adapted
from flow-based sampling) over binary masks `m ∈ {0,1}^n`, carrying an
ε-convergence guarantee.

This repository contains a complete, runnable implementation of every in-scope
algorithm, model, dataset, baseline, and experiment from the paper.

---

## 1. Scope

**In scope**

| Paper item | Driver |
|---|---|
| Figure 1 — trivial solutions of RCS (Eq. (3) / Eq. (4)) | `experiments/figure1_trivial.py`, `experiments/appendix_c3.py` |
| §5.1 / Table 1 — preliminary superiority on MNIST-S | `experiments/table1_prelim.py` |
| §5.2 / Tables 2–3 — comparison with competitors | `experiments/table2_table3_compare.py` |
| §5.3 / Figure 2 — robustness against imperfect supervision | `experiments/figure2_robustness.py` |
| Appendix E.2 — 50 % symmetric label noise | `experiments/figure2_robustness.py` |
| Appendix E.3 / Table 8 — optimized sizes under imperfect supervision | `experiments/table8_sizes.py` |
| §6 / Table 9 (Appendix E.4) — `T`-sweep | `experiments/table9_search_times.py` |
| §6 / Table 5 — mask initialization | `experiments/table5_init.py` |
| §6 / Table 6 — cross-architecture (ViT-small, WideResNet) | `experiments/table6_cross_arch.py` |

**Explicitly OUT of scope** (no driver invokes these; enforced in
`experiments/__init__.py::OUT_OF_SCOPE` and asserted by its self-test):

- §5.4 — ImageNet-1k experiments
- Appendix E.5 — continual learning
- Appendix E.6 — streaming coreset selection

---

## 2. Installation

```bash
# 1) virtual environment
python -m venv .venv && source .venv/bin/activate     # Python 3.9–3.10

# 2) PyTorch matching your CUDA version (install FIRST, see pytorch.org)
#    e.g. CUDA 11.8:
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

# 3) remaining dependencies
pip install -r lbcs_repro/requirements.txt
```

All datasets are downloaded automatically through `torchvision`
(MNIST, Fashion-MNIST, SVHN, CIFAR-10). **No Kaggle account and no API keys are
required.** Set `LBCS_DATA_ROOT=/path/to/data` to relocate the dataset cache.

CPU fallback is supported for smoke tests; the paper's §5.1/§5.2 runs used
NVIDIA GTX 3090 GPUs.

---

## 3. Quick start (smoke tests, CPU-friendly)

```bash
# run every offline self-test of the core machinery
python -m lbcs_repro.lbcs.masks
python -m lbcs_repro.lbcs.discretize
python -m lbcs_repro.lbcs.lexiflow
python -m lbcs_repro.lbcs.lexicographic
python -m lbcs_repro.data.datasets
python -m lbcs_repro.utils.metrics

# experiment drivers, dry/offline self-tests (no GPU, no training)
python -m lbcs_repro.experiments.figure1_trivial --selftest
python -m lbcs_repro.experiments.table1_prelim --selftest
python -m lbcs_repro.experiments.table2_table3_compare --selftest
python -m lbcs_repro.experiments.figure2_robustness --selftest
python -m lbcs_repro.experiments.table9_search_times --selftest
python -m lbcs_repro.main --list
```

Every driver also accepts `--smoke` for a tiny end-to-end run
(smaller `n`, fewer `T`/epochs/repeats) before committing to the full protocol.

---

## 4. Reproducing the experiments

The primary entry point is `main.py`; `scripts/run_all.py` orchestrates the whole
suite. Both deep-merge `configs/default.yaml` with a section overlay
(`configs/figure1.yaml`, `section5_1.yaml`, `section5_2.yaml`, `section5_3.yaml`,
`section6.yaml`).

```bash
# dispatch one experiment (config name or explicit path)
python -m lbcs_repro.main --experiment figure1
python -m lbcs_repro.main --experiment table1  --config configs/section5_1.yaml
python -m lbcs_repro.main --experiment table2
python -m lbcs_repro.main --experiment figure2
python -m lbcs_repro.main --experiment table8
python -m lbcs_repro.main --experiment table9
python -m lbcs_repro.main --experiment table5
python -m lbcs_repro.main --experiment table6

# override any nested config key from the CLI
python -m lbcs_repro.main --experiment table1 --set table1.repeats=1 --set table1.T=20

# run groups / everything in scope
python -m lbcs_repro.scripts.run_all --group core
python -m lbcs_repro.scripts.run_all --all --output-root results
```

### 4.1 Figure 1 — trivial solutions (`experiments/figure1_trivial.py`)

Reproduces the two failure modes of the trivial RCS formulations:

- **Eq. (3)**: `min_m f1(m)  s.t.  θ(m) ∈ argmin_θ L(m,θ)` — `f1` decreases while
  `f2` stays pinned near the predefined `k` (*fixed-size* failure mode).
- **Eq. (4)**: `min_m (1−λ) f1(m) + λ f2(m)` with `λ = 0.5` — `f2` collapses to
  near-zero while `f1` remains large (*over-minimization* failure mode).

Paper-stated settings (Appendix C.3 / Addendum): MNIST-S (1000 random MNIST
samples) or a random MNIST subset, ConvNet from Zhou et al. (2022), inner loop
**SGD `lr=0.1`, momentum `0.9`, 100 epochs**, outer loop **Adam `lr=2.5`,
cosine scheduler, `T=1000`**.

```bash
python -m lbcs_repro.experiments.figure1_trivial --paper
python -m lbcs_repro.experiments.appendix_c3 --paper       # settings + gradient analysis
```

Artifacts: `results/figure1/figure1.png`, `figure1_curves.json`, `figure1_summary.txt`.

### 4.2 §5.1 / Table 1 — preliminary superiority (`experiments/table1_prelim.py`)

MNIST-S, ConvNet, `k ∈ {200, 400}`, `ε ∈ {0.2, 0.3, 0.4}`, **20 repeats**;
reports mean ± std of `f1(m)` and `f2(m)`. Inner loop for §5.2/§5.1 uses
**Adam `lr=0.001`**.

```bash
python -m lbcs_repro.experiments.table1_prelim --paper
```

Expected direction: achieved `f1` and `f2` both lower than their initial values;
larger `ε` ⇒ smaller average `f2` (and non-decreasing average `f1`).

### 4.3 §5.2 / Tables 2–3 — competitor comparison (`experiments/table2_table3_compare.py`)

F-MNIST, SVHN, CIFAR-10; `k ∈ {1000, 2000, 3000, 4000}`; `ε = 0.2`, `T = 500`,
**10 repeats**. Baselines: Uniform, EL2N, GraNd, Influential, Moderate, CCS,
Probabilistic. **LBCS is the only method that also minimizes the coreset size**;
the others produce exactly `k` examples.

Target models after selection: F-MNIST → LeNet (Adam `lr=0.001`, 100 epochs);
SVHN → target CNN (Adam `lr=0.001`, 100 epochs); CIFAR-10 → ResNet-18
(SGD `lr=0.1`, momentum `0.9`, cosine, 200 epochs).

Table 3 re-uses the LBCS-achieved size for every baseline (matched-size
comparison).

```bash
python -m lbcs_repro.experiments.table2_table3_compare --paper
```

### 4.4 §5.3 / Figure 2 + Tables 8 — imperfect supervision

```bash
# accuracy comparison under 30 % / 50 % symmetric label noise and imbalance ρ = 0.01
python -m lbcs_repro.experiments.figure2_robustness --paper

# optimized coreset sizes under the same conditions (Appendix E.3)
python -m lbcs_repro.experiments.table8_sizes --paper
```

F-MNIST, `k ∈ {1000,2000,3000,4000}`, `ε=0.2`, `T=500`, 10 repeats. Label noise is
injected **only into the training split**; the test split stays clean. Class
imbalance uses the exponential scheme of `imbalance_cifar.py` with ratio `0.01`,
again training-only.

### 4.5 §6 ablations

```bash
# Table 9: T-sweep on F-MNIST, T in {100,200,300,500,800,1500,2000}
python -m lbcs_repro.experiments.table9_search_times --paper

# Table 5: LBCS vs LBCS + Moderate mask initialization
python -m lbcs_repro.experiments.table5_init --paper

# Table 6: SVHN cross-architecture targets (ViT-small, WideResNet)
python -m lbcs_repro.experiments.table6_cross_arch --paper
```

---

## 5. Code map

```
lbcs_repro/
  main.py                  # CLI entry point (config merge, dispatch, logging, seeds)
  scripts/run_all.py       # orchestrator for the whole in-scope suite
  lbcs/
    masks.py               # binary m ∈ {0,1}^n, relaxed masks, grouping, ||m||_0
    objectives.py          # f1(m), f2(m), L(m,θ), MaskObjectiveEvaluator + cache
    lexicographic.py       # Definition 1 relations, practical relations, F_H tracker
    lexiflow.py            # Algorithm 2 randomized direct search
    discretize.py          # clamp to [-1,1], project [-1,0)->0, [0,1]->1
    bilevel.py             # Algorithm 1 LBCS loop + InnerTrainer/TrainInfo/ModelStateBank
    acceleration.py        # warm start, sparsity, grouped masks (§3.2)
  models/                  # ConvNet, LeNet, SVHN CNN, CIFAR CNN, ResNet-18, ViT-small, WideResNet
  data/
    datasets.py            # MNIST(-S), F-MNIST, SVHN, CIFAR-10 via torchvision
    mnist_s.py             # 1000 random MNIST training samples (§5.1/Figure 1)
    robustness.py          # symmetric label noise (30 %/50 %) + exponential imbalance (0.01)
  baselines/               # Uniform, EL2N, GraNd, Influential, Moderate, CCS,
                           # Probabilistic (Zhou et al. 2022), Eq.(3)/Eq.(4) trivial
  experiments/             # one driver per paper artifact (see table above)
  utils/                   # metrics, seeding, logging, checkpointing
  configs/                 # default + per-section YAML overlays
  requirements.txt
```

### Algorithm 1 (LBCS)
Require network `θ`, dataset `D`, predefined size `k`, compromise `ε`.
Initialize binary masks with `||m||_0 = k`; for `t = 1..T`: train the inner loop to
obtain `θ(m)` (`θ(m) ← argmin_θ L(m,θ)`), then update masks by lexicographic
optimization (Algorithm 2); output the final mask.

### Algorithm 2 (LexiFlow)
Randomized direct search on the unit sphere with practical lexicographic
acceptance: sample `u ∈ S`, evaluate `m ± δu`, accept via the *update* procedure
against thresholds `F_H = [f̃1*, f̃2*]`; increment the stagnation counter on
rejection; shrink `δ ← δ·sqrt((t'+1)/(t+1))`; random-restart when `δ < δ_lower`.
Mask values are clamped to `[-1,1]` during search and projected to `{0,1}` at the
end.

---

## 6. Suggested (non-paper) defaults

The paper does not numerically specify some values. They are centralized in
`configs/default.yaml` and labelled `# SUGGESTED`; change them there rather than
in algorithm code.

| Parameter | Suggested default |
|---|---|
| `δ_init` | `0.1` |
| `δ_lower` | `1e-3` |
| `T` (when a section omits it) | `500` |
| batch size | `128` (eval `256`) |
| weight decay | `0.0` (non-ImageNet) |
| inner-loop epochs (when unspecified) | `100` |
| Figure 1 `k` | `200` |
| Probabilistic Adam `β1, β2, eps`, `s_min`, `s_max` | `0.9, 0.999, 1e-8, 1e-3, 1-1e-3` |

Exact network widths (Appendix D.2 Table 7) are not recoverable from the paper
PDF; every architecture field is exposed as an overridable `SUGGESTED` default in
its `*Config` dataclass (`models/*.py`) and can be set from YAML.

---

## 7. Reporting protocol

Mean ± standard deviation over the paper's repeat counts: **20** for §5.1 and
**10** for §5.2/§5.3/§6. Seeds are derived per repeat deterministically
(`utils/seed.py::resolve_seed`) and logged. Each driver writes JSON, CSV, TXT and
JSONL artifacts plus a `*_checks.json` recording whether the paper's qualitative
directions were observed.

---

## 8. Validation

```bash
# core unit checks (mask size, projection rules, lexicographic relations,
# threshold updates, restart / dynamic step-size bookkeeping)
python -m lbcs_repro.lbcs.masks
python -m lbcs_repro.lbcs.lexicographic
python -m lbcs_repro.lbcs.discretize
python -m lbcs_repro.lbcs.lexiflow
python -m lbcs_repro.utils.metrics
python -m lbcs_repro.utils.seed
python -m lbcs_repro.utils.checkpoint

# driver self-tests (offline, synthetic objectives)
python -m lbcs_repro.experiments.figure1_trivial --selftest
python -m lbcs_repro.experiments.table2_table3_compare --selftest
python -m lbcs_repro.experiments.figure2_robustness --selftest
```

Then compare produced artifacts under `results/` against the paper's reported
trends and (where given) numeric anchors embedded as `paper_reference` /
`PAPER_TABLE*` constants in the drivers.

---

## 9. Notes

- Out-of-scope sections (§5.4 ImageNet-1k, Appendix E.5 continual learning,
  Appendix E.6 streaming) are intentionally absent; `OUT_OF_SCOPE` in the
  experiment registry documents and enforces this.
- PyTorch is treated as a soft dependency in the core mask/lexicographic logic so
  mask-only unit tests run in a NumPy-only environment.
- All randomness is seeded; datasets load deterministically and test loaders use
  `shuffle=False` so repeated runs are comparable.
