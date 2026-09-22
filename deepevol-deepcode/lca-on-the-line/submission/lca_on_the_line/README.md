# LCA-on-the-Line

Reproduction of **"LCA-on-the-Line: In-Distribution Taxonomic Distance (LCA) Predicts Out-of-Distribution Generalization"**.

The paper introduces the **Lowest Common Ancestor (LCA)** distance measured on a class hierarchy
(the ImageNet WordNet taxonomy, or a *latent* K-means hierarchy learned from a model's own features)
as a **unified in-distribution (ID) metric** that strongly predicts **out-of-distribution (OOD)**
top-1 / top-5 accuracy — for both classic vision models (VMs) and vision-language models (VLMs).

This repository implements:

| Paper artifact | Implementation | Entry point |
|---|---|---|
| §2, §D.2.1 — LCA distance `D_LCA^I`, `D_LCA^P`, info content `I(y)=log|L|-log|L(y)|` | `src/hierarchy/{wordnet,info_content,lca,lca_matrix}.py` | `tests/test_lca_sanity.py` |
| §2, §D.3 — dataset-level `D_LCA`, `D_ELCA` | `src/metrics/lca_metric.py` | — |
| §D.1 — R², PEA, KEN, SPE, MAE, min-max/probit fits | `src/metrics/correlation.py` | — |
| §C, Table 7 — Gaussian-mixture simulation | `src/simulation/simulated_lca.py` | `python -m src.simulation.simulated_lca` |
| §4, Appendix A — 36 torchvision VMs + 39 VLMs | `src/models/{vm_zoo,vlm_zoo}.py` | — |
| §4 — ImageNet-1k + ImgN-v2/S/R/A/ObjectNet loaders | `src/data/{imagenet,ood_datasets}.py` | — |
| §4.1, Tables 1/2/8, Fig 1/5 — 75-model score table | `src/eval/evaluate_models.py` | `scripts/run_correlation.py` |
| §4.2, Table 3 — OOD prediction + baselines (ID Top-1, AC, Aline-D/S) | `src/eval/ood_prediction.py` | `scripts/run_ood_prediction.py` |
| §4.3.1, §E.1, Table 4 — latent K-means hierarchies | `src/hierarchy/latent_kmeans.py` | `scripts/run_latent_hierarchy.py` |
| §4.3.2, §E.2/§E.3/§E.5, Algorithm 1 — soft LCA labels + linear probe | `src/alignment/{soft_loss,linear_probe}.py` | `scripts/run_soft_label_probe.py` |
| §4.3.3, Table 14 — taxonomy-aware prompt engineering | `src/alignment/prompt_engineering.py` | `scripts/run_prompt_eval.py` |

---

## 1. Installation

Python 3.9+ is required.

```bash
git clone https://github.com/openai/CLIP.git   # or let pip do it (see requirements.txt)
pip install -r requirements.txt
```

`requirements.txt` groups the dependencies by subsystem, so the **metric / hierarchy / simulation**
stack (useful on its own for sanity checks) can be installed without the heavy VLM extras.

No API keys are needed: every checkpoint is public (torchvision, HuggingFace, OpenAI CLIP, OpenCLIP).

### Data

Download the datasets you intend to evaluate (or let the loaders fetch the HuggingFace copies):

| Dataset | Source used by `src/data/ood_datasets.py` |
|---|---|
| ImageNet-1k (ID) | HuggingFace `imagenet-1k` (`trust_remote_code=True`) |
| ImageNet-v2 MatchedFrequency | HF `vaishaal/ImageNetV2`, commit `d626240` |
| ImageNet-Sketch | HF `songweig/imagenet_sketch` |
| ImageNet-R | local ImageFolder (wnid-ordered classes) |
| ImageNet-A | official `.npy` arrays (`NpyClassDataset`) |
| ObjectNet | local folder + ObjectNet→ImageNet mapping file |
| WordNet tree | `imagenet_fiveai.csv` from [jvlmdr/hiercls](https://github.com/jvlmdr/hiercls) |

If `imagenet_fiveai.csv` is missing, the hierarchy falls back to nltk hypernym chains and finally to a
deterministic synthetic tree (`allow_synthetic=True`) so that offline runs and tests still work.

---

## 2. Configuration

All paths and hyperparameters live in [`configs/config.yaml`](configs/config.yaml) and are overridable
by CLI flags. The most important keys:

```yaml
dataset_root: /path/to/imagenet            # ID data (or null -> HuggingFace)
ood_roots: {v2: null, s: null, r: /path/I-R, a: /path/I-A, objectnet: /path/ObjectNet}
hierarchy: {csv_path: /path/imagenet_fiveai.csv, allow_synthetic: true, num_classes: 1000}
cache_dir: cache/outputs                   # <model>__<dataset>.npz feature/logit cache
results_dir: results

latent_hierarchy: {max_level: 9, base_level: 10, init: k-means++, seed: 0}
alignment: {lambda_weight: 0.03, temperature: 25.0, learning_rate: 0.001,
            batch_size: 1024, epochs: 50, weight_decay: 0.05, warmup_lr: 1e-5}
prompt_engineering: {encoder: CLIP_ViT32, max_ancestors: 2, templates: [...]}
```

Feature caching to `cache_dir` is **strongly recommended**: evaluating all 75 models is expensive, and
every downstream script (`run_ood_prediction`, `run_latent_hierarchy`, `run_soft_label_probe`) reuses
the cached `features` / `logits` / `targets` `.npz` files instead of re-running the models.

---

## 3. Running the reproduction phases

### Phase A — metric sanity checks (fast, no data needed)

```bash
python tests/test_lca_sanity.py            # standalone runner (33 checks)
# or
pytest tests/test_lca_sanity.py -q
```

Verifies: zero diagonal of the LCA distance matrix, ones on the diagonal of `reverse_LCA = 1 - M`,
symmetry, information-content monotonicity, the closed forms `D_LCA^I = I(y) - I(LCA)` and
`D_LCA^P`, dataset-level `D_LCA`/`D_ELCA`, correlation metrics, latent-hierarchy properties and the
Algorithm-1 soft loss.

### Phase G (cheap) — Appendix C / Table 7 simulation

```bash
python -m src.simulation.simulated_lca --num-trials 100 --num-samples 10000
```

Expected: model `f` (uses the causal feature `x1`) has ID error ≈ 0.159, ID LCA ≈ 1.005 and OOD error
≈ 0.320, while model `g` (confounder `x2`) has ID error ≈ 0.0 but OOD error ≈ 0.75 and ID LCA ≈ 2.0 —
i.e. lower LCA ⇒ better OOD.

### Phase B — Table 2 / Figures 1 & 5 (LCA-on-the-Line correlations)

```bash
python scripts/run_correlation.py --config configs/config.yaml --overwrite-cache
```

Produces `results/scores.json`, `results/table2_correlations.json`, `results/success_summary.json`
and the figures `figure1_lca_on_the_line.png`, `figure5_lca_error_fit.png`.

Reference points (paper): with **ID LCA** as predictor, ImageNet-S R² = 0.816 / PEA = 0.903,
ImgN-R 0.779 / 0.883, ImgN-A 0.704 / 0.839, ObjectNet 0.915 / 0.956; ImgN-v2 is the weak case
(R² = 0.339 / PEA = 0.582). The **ID Top-1** baseline collapses on the severe-shift sets
(R² ≈ 0.075 / 0.020 / 0.009). The script prints PASS/WARN lines for each of these gates.

### Phase C — Table 3 (OOD prediction + baselines)

```bash
python scripts/run_ood_prediction.py --config configs/config.yaml --scores-json results/scores.json
```

Expected MAE of the predicted OOD Top-1 (ID LCA predictor): ImgN-v2 0.162, ImgN-S 0.093,
ImgN-R 0.114, ImgN-A 0.103, ObjectNet 0.048 — beating ID Top-1 (0.230 / 0.277 / 0.192 / 0.178 on the
four severe-shift sets), Average Confidence (AC), and Aline-D / Aline-S.

### Phase D — Table 4 (latent hierarchies)

```bash
python scripts/run_latent_hierarchy.py --config configs/config.yaml
```

Builds one latent hierarchy per source model (K-means on per-class mean features, 9 levels with `2^i`
centers, base level 10) and reports mean PEA of ID-LCA-vs-OOD-Top-1 across hierarchies:
v2 0.815, S 0.773, R 0.712, A 0.662, ObjectNet 0.930.

### Phase E — Tables 5/6/9/10 and Figure 8 (taxonomy soft labels)

```bash
python scripts/run_soft_label_probe.py --config configs/config.yaml
```

Trains linear probes on frozen features (AdamW, lr 1e-3, batch 1024, cosine schedule with linear
warm-up, 50 epochs) with CE-only vs `CE + λ·L_soft_lca` (`λ=0.03`, `T=25`) and sweeps the weight
interpolation `W_interp = α·W_ce + (1-α)·W_ce+soft` to report the *no-ID-drop* and *pro-OOD*
operating points. Example paper numbers: ResNet50 on ImgN-R 36.5 → 42.5, ImgN-A 10.3 → 16.2;
ResNet-18 + MnasNet latent hierarchy on ImgN-S 19.7 → 20.2, WordNet 19.7 → 21.2.

### Phase F — Table 14 (prompt engineering)

```bash
python scripts/run_prompt_eval.py --config configs/config.yaml
```

CLIP-ViT32 zero-shot Top-1 on ImageNet: Baseline `"<class>"` 0.589 → Taxonomy Parent
`"<class>, which is a type of <parent>, which is a type of <grandparent>"` 0.626. Stack Parent and
Shuffle Parent must be worse than the taxonomy template, and test-time CE must decrease.

---

## 4. Repository layout

```
lca_on_the_line/
├── configs/config.yaml                 # paths, hyperparameters, model registries, paper references
├── src/
│   ├── hierarchy/                      # WordNet parsing, info content, D_LCA, matrix processing,
│   │                                   #   latent K-means hierarchies
│   ├── metrics/                        # dataset-level LCA / ELCA + correlation & regression suite
│   ├── models/                         # 36 torchvision VMs (vm_zoo) + 39 VLMs (vlm_zoo)
│   ├── data/                           # ImageNet-1k and ImgN-v2/S/R/A/ObjectNet loaders
│   ├── eval/                           # 75-model evaluation driver + OOD prediction baselines
│   ├── alignment/                      # Algorithm-1 soft loss, linear probing, prompt engineering
│   └── simulation/                     # Appendix C Gaussian-mixture study
├── scripts/                            # one runner per paper table/figure
├── tests/test_lca_sanity.py            # Phase A sanity suite (also runnable via pytest)
├── requirements.txt
└── README.md
```

Every `src` package uses lazy PEP-562 `__getattr__` re-exports, so importing a package never pulls in
torch / torchvision / open_clip / CLIP / `datasets` unless a heavy symbol is actually touched.

---

## 5. Key definitions implemented (paper notation)

* **Information content** (uniform leaf distribution, §D.2.1):
  `I(y) = log|L| - log|L(y)| = -log p(y)`, with `p(node) = |L(node)| / |L|`.
* **Depth score**: `P(x)` = node depth in the tree.
* **Pairwise distances** (§2):
  `D_LCA^I(y', y) = I(y) - I(N_LCA(y, y'))`,
  `D_LCA^P(y', y) = (P(y) - P(N_LCA)) + (P(y') - P(N_LCA))`.
  These are **distances** (zero diagonal, smaller = closer), not similarities.
* **Dataset-level metrics** (§2, §D.3), over misclassified samples only:
  `D_LCA(model, M) = (1/n) Σ_i D_LCA(ŷ_i, y_i)`,
  `D_ELCA(model, M) = (1/(n·K)) Σ_i Σ_k p̂_{k,i} · D_LCA(k, y_i)`.
* **Matrix processing** (§E.2): for WordNet use the raw `M`; for latent hierarchies invert first
  (`max(M) - M`), then raise to the temperature `M ** T`, then MinMax-scale to `[0, 1]`.
  The alignment indicator is `reverse_LCA_matrix = 1 - M_LCA`.
* **Algorithm 1** (§E.2): `L = λ · CE(logits, y) + L_soft`, with
  `L_soft = -mean(reverse_LCA_matrix[y] · log softmax(logits))` (a BCE variant is also provided);
  `λ = 0.03`, `T = 25`.

### Notes on conventions

* `D_LCA`/`R²`/`PEA` are reported as **absolute values** in the paper's tables.
* The linear fit uses **min-max** scaling because LCA is not in `[0, 1]`; **probit** is reserved for
  accuracy-based baselines (ID Top-1, AC, Aline), matching the referenced literature.
* **ELCA must not be compared across modalities** — it is sensitive to logit temperature.
* LCA is deliberately ineffective on ID-like sets such as ImageNet-v2; the paper only claims strong
  prediction under severe distribution shift (S/R/A/ObjectNet).

### Defaults chosen where the paper is silent

AdamW `weight_decay = 0.05`; interpolation α grid 0.0 → 1.0 step 0.1; K-means `init='k-means++'` with a
fixed seed and `n_init=10`; warm-up = 5 % of the total steps with `warmup_lr = 1e-5`; probit transform
for accuracy baselines; ObjectNet classes map to the *first* compatible ImageNet class.

---

## 6. Quick smoke test without any data or checkpoints

```bash
python tests/test_lca_sanity.py                                   # Phase A
python -m src.simulation.simulated_lca --num-trials 20             # Table 7 (fast)
python scripts/run_correlation.py --allow-synthetic --no-figures   # pipeline plumbing only
python scripts/run_latent_hierarchy.py --allow-synthetic
python scripts/run_soft_label_probe.py --allow-synthetic
python scripts/run_prompt_eval.py --synthetic
```

`--allow-synthetic` / `--synthetic` modes exist purely so the pipeline can be exercised offline; the
numbers they produce are **not** paper-valid.

---

## 7. Success criteria

* LCA correlations on the severe-shift datasets exceed **0.7** for both R² and PEA
  (`success_criteria.min_r2_severe_shift`, `min_pea_severe_shift` in `configs/config.yaml`);
* ID Top-1 baselines fail to unify VM and VLM trends (R² collapses on S/R/A);
* the LCA-based OOD predictor beats **ID Top-1, AC, Aline-D and Aline-S** in MAE on S/R/A/ObjectNet.
