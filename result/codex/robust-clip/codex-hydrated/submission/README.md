# Robust CLIP — reproduction

Reproduction of

> **Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust
> Large Vision-Language Models**
> Christian Schlarmann, Naman Deep Singh, Francesco Croce, Matthias Hein — ICML 2024

The paper's core contribution is **FARE** (*Fine-tuning for Adversarially Robust
Embeddings*): an **unsupervised** adversarial fine-tuning scheme for the CLIP
vision encoder that keeps the fine-tuned embedding close to the embedding of the
original CLIP model. Because the embedding is preserved, the robust encoder can
be dropped into LLaVA-1.5 and OpenFlamingo **without retraining anything**, and
the robustness transfers to every downstream task (zero-shot classification,
captioning, VQA, POPE, SQA, jailbreaking).

This repository contains a from-scratch implementation of

* the FARE loss and its adversarial fine-tuning procedure (2 epochs on ImageNet,
  10 PGD steps, ℓ∞ radii 2/255 and 4/255),
* the supervised TeCoA baseline of Mao et al. (2023) trained with the paper's
  hyper-parameters,
* the attack machinery (APGD in the ℓ∞ threat model, the half-precision /
  single-precision ensemble pipeline for LVLMs, and the stealthy targeted
  attacks),
* the LLaVA-1.5 7B / OpenFlamingo-9B integrations with a swappable OpenCLIP
  vision encoder, and
* all evaluation protocols of the main paper (zero-shot classification,
  captioning, VQA, POPE, ScienceQA-I, jailbreaking).

Nothing from the authors' code base (the repository on the task's blacklist) was
used; the implementation follows the paper text, its appendix and the
addendum, plus the public upstream projects that the addendum names (OpenCLIP,
LLaVA, OpenFlamingo, the APGD of `fra31/robust-finetuning`, and the
jailbreaking attack of Qi et al.).

---

## 1. What is reproduced, and where

| Paper element | Code | Status |
|---|---|---|
| **FARE loss**, Eq. (3), Sec. 3.3 | [`robust_clip/training/losses.py`](robust_clip/training/losses.py) `FARELoss` | implemented |
| **TeCoA loss**, Eq. (2), Sec. 3.2 | `TeCoALoss` (same file) | implemented |
| PGD inner maximisation, App. B.1 + addendum | [`robust_clip/training/pgd.py`](robust_clip/training/pgd.py) | implemented |
| Fine-tuning recipe (2 epochs, 10 PGD steps, `1/255` step, AdamW, cosine + 7 % warm-up, batch 128) | [`robust_clip/training/train.py`](robust_clip/training/train.py), [`configs/`](configs) | implemented |
| Theorem 3.1 (embedding ↔ cosine bound, App. A) | [`robust_clip/theory.py`](robust_clip/theory.py) + `test_theorem_3_1_holds` | implemented + verified numerically |
| **Table 1** — LLaVA/OpenFlamingo captioning + VQA, clean & adversarial | [`robust_clip/eval/captioning.py`](robust_clip/eval/captioning.py), [`vqa.py`](robust_clip/eval/vqa.py) | implemented |
| **Table 2** — transfer attacks between LVLMs | [`robust_clip/eval/transfer_attack.py`](robust_clip/eval/transfer_attack.py) | implemented |
| **Table 3 / Fig. 3** — stealthy targeted attacks on LLaVA | [`robust_clip/eval/targeted_attack.py`](robust_clip/eval/targeted_attack.py) | implemented |
| **Table 4** — zero-shot classification (ImageNet + 13 datasets) | [`robust_clip/eval/zero_shot.py`](robust_clip/eval/zero_shot.py), [`zs_datasets.py`](robust_clip/eval/zs_datasets.py) | implemented |
| **Fig. 1** — radar plot of the clean performance; **Fig. 2/3** qualitative outputs | [`robust_clip/eval/figures.py`](robust_clip/eval/figures.py), qualitative strings stored in the result JSONs | implemented |
| **Table 5** — POPE hallucination F1 | [`robust_clip/eval/pope.py`](robust_clip/eval/pope.py) | implemented |
| **Table 6** — ScienceQA-I (chain of thought) | [`robust_clip/eval/sqa.py`](robust_clip/eval/sqa.py) | implemented |
| **Table 7** — jailbreaking attacks | [`robust_clip/eval/jailbreak.py`](robust_clip/eval/jailbreak.py) | implemented (harmfulness judged by a keyword heuristic; the paper uses a human annotator) |
| Attack pipeline of Sec. 4.1 / App. B.6 (half precision → single precision, thresholds, targeted "maybe"/"Word") | [`robust_clip/attacks/lvlm_attack.py`](robust_clip/attacks/lvlm_attack.py) | implemented |
| APGD with CE / targeted DLR (100 iterations) | [`robust_clip/attacks/apgd.py`](robust_clip/attacks/apgd.py) | implemented |
| Tab. 14 / App. C.4 embedding losses (appendix, out of scope) | [`robust_clip/eval/embedding_loss.py`](robust_clip/eval/embedding_loss.py) | implemented for completeness |
| App. B.3/B.4/B.5 ablations (LR/WD, loss, TeCoA checkpoint) | same training script with [`configs/ablation_vitb32.json`](configs/ablation_vitb32.json) | supported |

---

## 2. Repository layout

```
robust_clip/
├── models.py                 # OpenCLIP loading: class token (φ) and penultimate patch tokens
├── theory.py                 # Theorem 3.1 / App. A: embedding distance bounds the cosine change
├── training/
│   ├── pgd.py                # PGD inner maximisation (elementwise-sign gradient normalisation,
│   │                         #   momentum 0.9, uniform random init, ball in pixel space)
│   ├── losses.py             # FARELoss (Eq. 3) and TeCoALoss (Eq. 2)
│   ├── data.py               # ImageNet via HuggingFace datasets or ImageFolder
│   └── train.py              # the fine-tuning entry point (all hyper-parameters of App. B.1)
├── attacks/
│   ├── apgd.py               # APGD / AutoAttack: CE, DLR, targeted DLR; ℓ∞ projector; integer quantisation
│   └── lvlm_attack.py        # ensemble attack pipeline (B.6) + stealthy targeted attacks (4.2)
├── lvlm/
│   ├── base.py               # LVLM interface (nll / generate / precision switching)
│   ├── llava.py              # LLaVA-1.5 7B with an OpenCLIP vision tower
│   ├── openflamingo.py       # OpenFlamingo-9B with a swappable vision encoder
│   └── factory.py            # backend selection
├── eval/
│   ├── zero_shot.py          # Table 4
│   ├── zs_datasets.py        # 14 datasets + CLIP_benchmark prompt templates
│   ├── captioning.py, vqa.py # Table 1
│   ├── transfer_attack.py    # Table 2
│   ├── targeted_attack.py    # Table 3
│   ├── pope.py, sqa.py, jailbreak.py
│   ├── embedding_loss.py     # App. C.4 (Table 14): clean / adversarial embedding losses
│   ├── figures.py            # Figure 1 (radar plot) + LaTeX tables from the result JSONs
│   ├── cider.py              # CIDEr-D
│   ├── metrics.py            # VQA accuracy, POPE F1
│   └── lvlm_data.py          # COCO / Flickr30k / VQAv2 / TextVQA loaders
├── utils/                    # seeding, devices, epsilon parsing, integer-grid quantisation
configs/                      # the exact hyper-parameters of the paper's training runs
scripts/                      # one shell script per experiment
tests/                        # smoke, encoder, LLaVA/OpenFlamingo, driver and attack-optimality tests
reproduce.sh                  # end-to-end driver
```

---

## 3. Method details as implemented

### 3.1 FARE (Eq. (3))

```
L_FARE(φ, x) = max_{||z − x||∞ ≤ ε} || φ(z) − φ_org(x) ||₂²
```

* `φ_org(x)` is the **original, frozen CLIP** embedding of the **clean** image
  (detached, see `FARELoss.reference_embedding`).
* only `φ(z)` — the encoder that is being fine-tuned, evaluated at the perturbed
  image — receives gradients.
* the loss is computed on the **class token** only, as in App. B.1
  (`--feature projected_class_token`, the ℓ₂-loss of the embedding that the
  zero-shot classifier of Eq. (1) uses; `--feature class_token` uses the
  pre-projection token instead). The appendix notes that using all tokens did not
  improve the results, so the default is the class token.
* the text encoder is never used and never updated (the method is unsupervised).
* because `L_FARE → 0` implies `φ(x) → φ_org(x)` for clean `x`, the robustness
  transfers to LLaVA/OpenFlamingo, where the projection layer and the language
  model stay frozen.

### 3.2 TeCoA (Eq. (2))

Supervised adversarial training of the zero-shot ImageNet classifier
`f_k(φ, x) = cos(φ(x), ψ(t_k))` with the text embeddings of the 1000 ImageNet
classes (`"A photo of a <class>."`). Following CLIP's zero-shot classifier, the
cosine logits are scaled by the CLIP logit scale (100) before the
cross-entropy; `--tecoa-temperature` in `TeCoALoss` exposes this.

### 3.3 Adversarial training / attack details

* 10 PGD steps for the inner maximisation, step size `1/255`, radii `2/255` and
  `4/255` (Sec. 4, App. B.1).
* element-wise sign gradient normalisation for ℓ∞, momentum 0.9, uniform random
  initialisation, and the ℓ∞ ball **around the non-normalised inputs** (the
  perturbations live in pixel space; the CLIP normalisation happens inside the
  model) — exactly the four properties listed in the addendum.
* AdamW (`β₁ = 0.9`, `β₂ = 0.95`), weight decay `1e-4`, effective batch size 128,
  peak LR `1e-5`, cosine decay with a linear warm-up to the peak at 7 % of the
  steps.
* half-precision attacks use **16-bit integer** perturbations, single-precision
  attacks **32-bit integer** ones (`robust_clip/utils/quant.py`), as the
  addendum requires.

### 3.4 LVLM integration

`OpenClipVisionTower` replaces LLaVA's HuggingFace CLIP tower and returns the
penultimate-block patch tokens of the OpenCLIP ViT (LLaVA-1.5's
`vision_feature_layer = -2`, `select_feature = 'patch'`). The tower emulates the
HuggingFace `CLIPVisionModel` output (`BaseModelOutputWithPooling.hidden_states`)
so that `LlavaForConditionalGeneration` works unchanged and the *only* change
with respect to vanilla LLaVA is the vision encoder — which is exactly the
intervention studied in the paper. OpenFlamingo consumes the full token
sequence of the ViT: `OpenClipFlamingoVisionEncoder` reproduces OpenFlamingo's
contract (`self.vision_encoder(vision_x)[1]` must yield the `B x N x width` token
sequence, no class token) and `patch_flamingo_for_attacks` removes the
`torch.no_grad()` that upstream wraps around the vision encoder — without it the
white-box attacks on the image would be impossible.

Vision inputs are kept in pixel space `[0, 1]` end-to-end; normalisation happens
inside the tower. This is what makes the attacks perturbations of the
*non-normalised* inputs.

### 3.5 Attack pipeline of Sec. 4.1 / App. B.6

`LVLMAttackPipeline` implements:

1. **Captioning**: APGD (100 iterations) at half precision against each of the
   five ground-truth captions. After every attack the CIDEr scores are computed;
   samples whose score dropped below the threshold (10 for COCO, 2 for
   Flickr30k) are not attacked any more. The worst score, the corresponding
   ground truth and its perturbation are remembered. Finally a single-precision
   attack (100 iterations) is run on the worst ground truth, initialised with
   the perturbation from the half-precision stage.
2. **VQA**: the same scheme with the five most frequent answers, a threshold of
   0, followed by targeted single-precision attacks with the target strings
   `"maybe"` (lower case) and `"Word"` (capitalised) starting from a clean
   perturbation initialisation. The `"Word"` attack is skipped for TextVQA.
3. The initial APGD step size is `ε` (Schlarmann & Hein, 2023) instead of the
   AutoAttack default of `2ε`.

The final metric per sample is the **worst case** over all attacks.

### 3.6 Stealthy targeted attacks (Sec. 4.2)

APGD with **10 000 iterations** (the appendix shows that 500 iterations only
reach a 59 % success rate at `2/255`), minimising the negative log-likelihood of
the target caption; an attack counts as successful if the target string appears
verbatim in the generated output. Six targets × 25 images × 2 radii reproduce
Table 3.

---

## 4. How to run

```bash
pip install -r requirements.txt          # Python ≥ 3.9, a GPU is strongly recommended

# 0. (optional) fetch the datasets in the layout the scripts expect
bash scripts/download_data.sh coco flickr vqa pope sqa jailbreak

# 1. fine-tune the four robust encoders (FARE^2/4, TeCoA^2/4) on ImageNet
#    (NGPU>1 launches torchrun/DDP; 8 GPUs give the paper's effective batch of 128)
IMAGENET_ROOT=/path/to/imagenet NGPU=8 bash scripts/train_fare_tecoa.sh

# 2. zero-shot classification (Table 4)
bash scripts/eval_zero_shot.sh

# 3. LLaVA captioning / VQA (Table 1) and targeted attacks (Table 3)
bash scripts/eval_lvlm_captioning.sh
bash scripts/eval_vqa.sh
bash scripts/eval_targeted.sh

# ... and the same tasks with OpenFlamingo-9B (rows 7-11 of Table 1)
bash scripts/eval_openflamingo.sh

# 4. POPE / SQA-I / jailbreaking (Tables 5–7)
bash scripts/eval_pope_sqa.sh
bash scripts/eval_jailbreak.sh

# 5. transfer attacks (Table 2) and the figure of the paper
bash scripts/eval_transfer.sh
python -m robust_clip.eval.figures \
    --results "CLIP=results/zero_shot/CLIP.json,results/captioning/CLIP_coco.json,results/vqa/CLIP_vqav2.json" \
    --out results/figure1.png
```

`bash reproduce.sh <stage>` runs the same stages (`train`, `zero_shot`,
`caption`, `vqa`, `targeted`, `other`, `all`).

Every entry point is a module with `--help`, e.g.

```bash
python -m robust_clip.training.train --help
python -m robust_clip.eval.zero_shot --help
```

### Cheap sanity checks (no GPU needed)

```bash
python tests/test_smoke.py       # or: pytest tests -q

# the FARE mechanism end-to-end on real images (16 CIFAR-10 images, CPU, ~4 min)
python scripts/quick_fare_check.py --device cpu

# end-to-end run on 2 synthetic classes with ViT-B/32 (≈40 s on CPU)
python -m robust_clip.training.train --method fare --arch ViT-B-32 \
    --imagenet-root /tmp/tiny_imagenet --epochs 1 --batch-size 2 \
    --max-steps-per-epoch 2 --num-workers 0 --device cpu --output-dir /tmp/out
```

---

## 5. Data and model requirements

| What | Where it comes from |
|---|---|
| CLIP weights | `open_clip` (`ViT-L-14`/`ViT-B-32`, `pretrained="openai"`) |
| ImageNet (training) | `datasets.load_dataset("imagenet-1k", trust_remote_code=True)` (addendum) or a local ImageFolder root |
| ImageNet / 13 zero-shot datasets | HuggingFace hub or a `CLIP_benchmark` style root (`--data-root`) |
| LLaVA-1.5 7B | `llava-hf/llava-1.5-7b-hf` (language model + projector); the vision encoder is replaced by OpenCLIP |
| OpenFlamingo-9B | `openflamingo/OpenFlamingo-9B-vitl-mpt7b` |
| COCO / Flickr30k / VQAv2 / TextVQA | original annotation files (`--data-root`) or the HuggingFace copies |
| POPE annotations | `RUCAIBox/POPE` (`coco_pope_{random,popular,adversarial}.json`) |
| ScienceQA | `derek-thomas/ScienceQA` (test split restricted to examples with an image) |
| Jailbreaking corpora | the CSVs linked in the addendum (`derogatory_corpus.csv`, `manual_harmful_instructions.csv`, `clean.jpeg`); downloaded automatically |

`scripts/download_data.sh <dataset>` fetches all of the above (except the
150 GB ImageNet, which it only loads through `datasets`).  LLaVA-1.5 **13B**
(App. C.3) only needs `--llava-path llava-hf/llava-1.5-13b-hf`.

No API keys are needed: every model and dataset is public.

---

## 6. Deviations, assumptions and known limitations

These are the places where the reproduction had to make a decision because the
paper/addendum do not fully specify the detail, or because a resource is not
available in this environment. None of them changes the qualitative claims of
the paper (FARE > TeCoA on clean accuracy everywhere; FARE ≥ TeCoA on
robustness; original CLIP completely non-robust).

1. **CIDEr document frequencies.** `cider.py` implements CIDEr-D from scratch
   (n-grams 1–4, Gaussian length penalty `σ = 6`) and is numerically **identical
   to `pycocoevalcap`** — `tests/test_cider.py` pins this down
   (`scale=10` reproduces `pycocoevalcap` exactly). The default `scale=1000`
   reports the metric in percent, the convention of the paper's tables (LLaVA-13B
   ≈ 119 ⇔ `pycocoevalcap` ≈ 1.19). The only free parameter is the tf-idf
   document frequency: it is computed from the evaluation corpus by default
   (an image counts once per n-gram, exactly as `pycocoevalcap` does with
   `df_mode="corpus"`), and `--cider-df` accepts the official COCO
   document-frequency file that the paper uses, which mainly rescales the
   absolute values.
2. **Prompts.** LLaVA uses the LLaVA-1.5 system prompt and the task prompts of
   Liu et al. (captioning, VQA, POPE, SQA); OpenFlamingo uses the
   `Question: … Short answer:`/plain-image prompts of its own evaluation
   protocol. The exact strings are in `robust_clip/lvlm/llava.py`
   (`TASK_QUESTIONS`) and `openflamingo.py` (`OPENFLAMINGO_PROMPTS`) so they can
   be adjusted without touching the rest of the code.
3. **The sixth targeted-attack image set.** The paper uses hand-picked stock
   photos of patients/syringes for target 6; the addendum explicitly exempts
   those images from reproduction, so `eval/targeted_attack.py` draws all 25
   images from COCO (`--images` allows supplying a custom set).
4. **Jailbreaking harmfulness.** The paper's criterion is human judgement. The
   code implements the attack faithfully and provides a transparent keyword
   heuristic (`harmfulness_judge`) plus a `judge` hook so that human/LLM
   annotations can be plugged in; the numbers in Table 7 are human-annotated.
5. **Vision-encoder resolution.** Following the addendum, LLaVA-1.5 runs with
   ViT-L/14 **@224** rather than the default 336, i.e. 256 visual tokens. The
   number of `<image>` placeholders is derived from the encoder's patch grid at
   runtime, so switching to 336 requires only `--image-size 336 --clip-arch
   ViT-L-14-336`.
6. **Half-precision quantisation.** The addendum's "16-bit ints for
   half-precision attacks, 32-bit ints for single-precision attacks" is
   implemented as rounding the perturbation onto the `{0…255}` integer grid
   with `int16`/`int32` (`utils/quant.py`).
7. **Dataset revisions.** The HuggingFace dataset ids in `zs_datasets.py` /
   `lvlm_data.py` are the closest public mirrors of the datasets used by the
   paper (which evaluates the original COCO/Flickr30k/CLIP_benchmark files); all
   of them can be replaced by a local `--data-root` without code changes.
8. **Environment.** This repository was developed and smoke-tested on a machine
   without a GPU (CPU/MPS only). The training and evaluation commands are
   written for the multi-GPU/GPU setting (bf16/fp16, batch size 128) and were
   *not* run to completion here (2 epochs of ViT-L/14 on ImageNet need GPUs); the
   components were verified individually instead: the FARE/TeCoA losses and
   their gradients on real images, the closed-form optimality of APGD, the
   attack pipeline with a stub LVLM, the metric implementations against their
   references, the LLaVA/OpenFlamingo integrations with tiny models, and the
   full zero-shot loop on real data (see `tests/` and the verification table
   above).
9. **CIFAR10/CIFAR100/STL-10 resolution.** App. B.10 says those three datasets
   are evaluated "at their respective original resolution". Feeding 32x32 images
   to a patch-14/patch-32 ViT, however, drops the *clean* accuracy of CLIP itself
   to ~50% (measured), which is incompatible with the clean numbers of Table 4,
   so the default is to resize every dataset to 224 (as ``CLIP_benchmark``
   does).  ``--native-resolution`` selects the literal reading; the positional
   embedding is interpolated at runtime in that case
   (`CLIPImageEncoder.interpolate_pos_embed_if_needed`), and the interpolation is
   always computed from the pristine weights, so switching resolutions is
   reversible.

Verified numbers of the pipeline (small-scale checks that were run here):

| Check | Result |
|---|---|
| CLIP ViT-B/32, zero-shot CIFAR-10, 128 images, 224px | **91.4 %** clean, matching the official `open_clip` preprocessing exactly |
| same, full-strength attack (100 iterations of APGD-CE **and** targeted APGD-DLR) | **0.0 % at 2/255 and 0.0 % at 4/255** (APGD-CE alone leaves 3.1 % / 9.4 %), i.e. *"the clean CLIP model is completely non-robust even at the small radius"* (Table 4) |
| FARE loss vs. its closed form; embedding distance decreases under 10 AdamW steps | exact / monotone decrease |
| LVLM attack pipeline (stub LVLM) | stays in the ℓ∞ ball and reduces the victim's score |
| `scripts/quick_fare_check.py` (16 CIFAR-10 images, 6 steps, eps=4/255, CPU) | see the table below |
| full chain `train.py` → checkpoint → `load_clip_encoder` → `zero_shot.py` on a 2-class CIFAR folder (4 FARE steps, CPU) | runs end to end, clean accuracy 88 % (89 % before training), robust 0 % |

`scripts/quick_fare_check.py` (identical budget for both methods, `--lr 1e-6`):

| Method | `E[L_clean]` | `E[L_adv]` | clean acc | robust acc |
|---|---|---|---|---|
| original CLIP | 0.00 | 141.34 | 88.3 % | 21.7 % |
| **FARE** (6 steps) | **2.67** | **129.98** | **85.0 %** | **25.0 %** |
| TeCoA (6 steps) | 4.87 | 135.44 | 86.7 % | 25.0 % |

Even at this toy scale the ordering matches Table 14 of the paper: FARE keeps the
clean embedding closest to the original CLIP *and* reduces the adversarial
embedding loss the most, whereas the supervised TeCoA baseline distorts the clean
embedding noticeably more while gaining no more robustness. (The absolute numbers
of the paper require the full 2-epoch ImageNet training; with 6 steps on 16
images only the direction is meaningful.)

---

## 7. Expected qualitative results

Reproducing the paper's trends means observing, after training the four
encoders:

* **Zero-shot classification (Table 4).** CLIP is best on clean data but drops
  to ≈0 % at `2/255`; FARE² keeps a clean average close to CLIP (≈67 % vs 73 %)
  while FARE⁴ is the most robust at both radii; TeCoA loses much more clean
  accuracy (≈60 % / 54 %).
* **LLaVA / OpenFlamingo (Table 1).** FARE beats the respective TeCoA model on
  clean *and* robust CIDEr/VQA accuracy; FARE² stays close to the original CLIP
  on clean inputs.
* **Targeted attacks (Table 3).** LLaVA-CLIP breaks 25/25 for every target;
  TeCoA⁴/FARE⁴ never break, and TeCoA²/FARE² only in a few cases at `4/255`.
* **POPE (Table 5) / SQA-I (Table 6).** CLIP best, FARE the closest robust
  model, TeCoA worst (most hallucinations, largest CoT drop).
* **Jailbreaking (Table 7).** Both TeCoA and FARE substantially reduce the
  harmful-output rate compared to the original CLIP encoder.

---

## 8. Tests

The test-suite (30 tests, no GPU; only the CIFAR-10 check and the two
integration tests touch the network / downloaded packages) covers every building
block of the reproduction:

`tests/test_smoke.py` (13 tests) checks:

* ε parsing and integer-grid quantisation,
* the PGD inner maximisation stays in the ℓ∞ ball and increases the objective,
* the FARE loss equals its closed form with respect to the *clean* reference
  embedding and decreases under a gradient step (this is Theorem 3.1's
  mechanism),
* TeCoA's logits equal the scaled cosine similarities,
* APGD breaks a toy classifier and respects the ball,
* the targeted/untargeted DLR losses and the runner-up target selection,
* the LR schedule (7 % linear warm-up, cosine decay),
* CIDEr prefers the matching caption, VQA accuracy and POPE F1,
* the full LVLM attack pipeline (half → single precision → targeted) stays in
  the ball and reduces the victim's score,
* Theorem 3.1 holds numerically (the right hand side really bounds the change of
  every cosine similarity).

`tests/test_lvlm_eval_drivers.py` drives the captioning and VQA evaluations with
a stub LVLM and checks that the prompts, the clean scoring, the full attack
pipeline, the stealthy **targeted** string attack (Sec. 4.2) and the metrics all
fit together.

`tests/test_attacks_optimality.py` compares APGD and the training-time PGD
against the **closed-form optimum** of the ℓ∞ attack on a linear model (the
optimum also has to respect the image box, which is how the paper defines the
feasible set).  APGD reaches the exact optimum on most samples and is within a
few percent on the rest, so the robustness numbers are not inflated by a weak
attack.

`tests/test_openflamingo_integration.py` builds a *tiny* Flamingo with the real
`open_flamingo` package (random OpenCLIP ViT-B/32 + `tiny-random-gpt2`) and
checks that the swapped-in encoder satisfies OpenFlamingo's
`(pooled, tokens)` contract, that the perceiver resampler consumes exactly those
tokens, and that gradients reach `vision_x` after
`patch_flamingo_for_attacks`.  (The language-model part of that test is skipped
when the installed `open_flamingo`/`transformers` versions are incompatible —
the released `open_flamingo` targets transformers ~4.30.)

`tests/test_cider.py` compares the captioning metric with
`pycocoevalcap` on a synthetic corpus and verifies that the default reporting
scale is percent.

`tests/test_llava_integration.py` builds a *tiny* LLaVA (randomly initialised
language model, randomly initialised OpenCLIP ViT-B/32, a word-level stub
tokenizer) and verifies the parts of the LLaVA integration that are easy to get
wrong and hard to notice:

* the OpenCLIP tower really is the module that `LlavaModel.forward` calls,
* it returns `hidden_states` in the HuggingFace layout (13 entries for 12
  blocks, `hidden_states[-2]` = CLS + 49 patch tokens), so LLaVA's own
  `vision_feature_layer = -2` / `select_feature = 'patch'` logic is reused and
  the projector receives `(49, hidden)` features,
* the number of `<image>` placeholders inserted by `tokenize_prompts` equals the
  number of patch tokens of the encoder, and
* `nll` is differentiable with respect to the input image (the prerequisite of
  every attack) and `generate` runs.

`tests/test_clip_encoder.py` locks down the two conventions that the whole
codebase depends on:

* `encode_image` on pixel-space `[0, 1]` images produces **exactly** the same
  embedding as the official `open_clip` transform (which normalises internally),
  so the ℓ∞ ball really is computed around the non-normalised inputs;
* interpolating the positional embedding for another resolution is reversible
  (224 → 32 → 224 gives bit-identical embeddings), and the patch-token count
  follows the patch grid.
