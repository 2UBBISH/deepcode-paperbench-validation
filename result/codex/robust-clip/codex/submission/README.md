# Robust CLIP — reproduction

Reproduction of

> **Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust
> Large Vision-Language Models**
> Christian Schlarmann\*, Naman Deep Singh\*, Francesco Croce, Matthias Hein — ICML 2024

The paper shows that the CLIP vision encoder which is frozen inside large
vision-language models (LLaVA-1.5, OpenFlamingo) can be made adversarially robust
by an **unsupervised** adversarial fine-tuning scheme (**FARE**), and that the
resulting encoder transfers that robustness to every down-stream task of the
LVLM without any retraining of the LVLM — while keeping the clean performance
close to the original CLIP model.

This repository contains a complete, self-contained implementation of

* the **FARE training objective** (Eq. (3)) and the supervised **TeCoA** baseline,
  including the ImageNet adversarial fine-tuning of the vision encoder
  (Sec. 3, App. B.1/B.3),
* the **attack suite** used in the paper: APGD (Croce & Hein, 2020), the PGD of
  the training and the targeted attacks, the precision-aware **attack ensemble**
  for LVLMs (Sec. 4.1, App. B.6), the **stealthy targeted attacks** (Sec. 4.2,
  App. B.8/B.9) and the **universal jailbreaking attack** of Qi et al. (2023)
  (Sec. 4.4),
* the **LLaVA-1.5 / OpenFlamingo wrappers** that plug a (robust) OpenCLIP vision
  tower into the frozen LVLM (Sec. 4.1, App. C.3, addendum),
* the **evaluation harness** for all main-body experiments: Table 1 (LVLM clean +
  robust performance), Table 2 (transfer attacks), Table 3 (targeted attacks),
  Table 4 (zero-shot classification + AutoAttack), Table 5 (POPE), Table 6
  (SQA-I) and Table 7 (jailbreaking),
* a full set of **unit / smoke tests** that validate the losses, the attacks, the
  metrics, the precision handling and the multimodal LLaVA input construction.

## 1. The method

### FARE (Eq. (3) of the paper)

The frozen original CLIP encoder is denoted by `phi_Org`, the fine-tuned one by
`phi_FT`.  For an image `x` the training loss is

```
L_FARE(x) = max_{||delta||_inf <= eps} || phi_FT(x + delta) - phi_Org(x) ||_2^2
            + lambda * || phi_FT(x) - phi_Org(x) ||_2^2
```

* the inner maximization is solved with **10 PGD steps** of step size `1/255`
  inside the `l_inf` ball of radius `eps` (2/255 or 4/255),
* the target of the regression is the embedding of the **original** CLIP model,
  hence the scheme is unsupervised (no labels, no text pairs),
* `lambda = 1` in all experiments, and the loss is computed on the **class token
  only** (App. B.1),
* squared `l_2` is used because it is a monotone function of the cosine
  similarity used by zero-shot classification and because it preserves the
  non-normalized embeddings (App. B.4, Theorem A.1, App. C.4).

Implementation: [`robust_clip/training/losses.py`](robust_clip/training/losses.py)
(`fare_loss`), the inner attack in
[`robust_clip/attacks/pgd.py`](robust_clip/attacks/pgd.py), the training loop in
[`robust_clip/training/adv_train.py`](robust_clip/training/adv_train.py).

### TeCoA (supervised baseline, Mao et al., 2023)

```
L_TeCoA(x) = max_{||delta||_inf <= eps} CE( phi_FT(x + delta) . psi(t) , y )
```

i.e. the usual image-text contrastive loss, but evaluated on adversarially
perturbed images.  It is the `--method tecoa` of the training script and serves
as the reference point of Tables 1–6.

### Embedding losses of App. C.4 (Table 14)

`L_clean(x) = ||phi_FT(x) - phi_Org(x)||_2^2` and
`L_adv(x) = max_{||z-x||_inf <= eps} ||phi_FT(z) - phi_Org(x)||_2^2`
(`clean_embedding_loss` / `adversarial_embedding_loss` in
[`robust_clip/training/losses.py`](robust_clip/training/losses.py),
evaluated by `scripts/eval_embedding_loss.py`).

## 2. Training the robust encoders (Sec. 3, App. B.1/B.3)

```bash
# FARE^4 vision encoder used by LLaVA (OpenAI CLIP ViT-L/14@224)
python scripts/train_clip.py --method fare --arch ViT-L-14 --pretrained openai \
    --eps 4/255 --epochs 2 --batch-size 128 --lr 1e-5 --wd 1e-4 \
    --pgd-steps 10 --pgd-alpha 1/255 --run-name FARE4-ViT-L-14-openai

# FARE^2
python scripts/train_clip.py --method fare --eps 2/255 ... --run-name FARE2-ViT-L-14-openai

# TeCoA baselines
python scripts/train_clip.py --method tecoa --eps 4/255 ... --run-name TeCoA4-ViT-L-14-openai

# vision encoder of OpenFlamingo (the LAION CLIP weights, 224x224)
python scripts/train_clip.py --method fare --pretrained laion2b_s32b_b82k --eps 4/255 ...

# ablation of App. B.3 with ViT-B/32
python scripts/train_clip.py --method fare --arch ViT-B-32 --eps 4/255 --run-name FARE4-ViT-B-32
```

Hyper-parameters (App. B.1/B.3, all implemented as defaults / CLI flags):

| setting | value |
| --- | --- |
| dataset | ImageNet, resolution 224×224 (`load_dataset("imagenet-1k", trust_remote_code=True)`) |
| epochs | 2 |
| inner attack | 10 PGD steps, step size 1/255, `eps ∈ {2/255, 4/255}` |
| optimizer | AdamW, `beta1 = 0.9`, `beta2 = 0.95` |
| learning rate | cosine decay, linear warmup to 1e-5 at 7 % of the steps |
| weight decay | 1e-4 |
| effective batch size | 128 (gradient accumulation via `--micro-batch-size`) |
| trainable | vision encoder only (the text tower stays frozen) |

The checkpoint contains the full CLIP state dict, so it can be loaded by
`--checkpoint` in all evaluation scripts (`robust_clip.models.load_clip`).

## 3. Attacks

| attack | where | details |
| --- | --- | --- |
| PGD (training, targeted) | `attacks/pgd.py` | normalized gradient + elementwise sign, momentum 0.9 (0 for the jailbreaking attack), uniform random init, `l_inf` ball around **non-normalized** inputs |
| APGD | `attacks/apgd.py` | APGD-CE and APGD-DLR with 100 iterations, momentum term and adaptive per-sample step size of Croce & Hein (2020); initial step size `eps` (App. B.6); targeted DLR uses the 9 closest classes (AutoAttack standard) |
| attack ensemble (LVLM) | `attacks/lvlm_attack.py` | ① half precision APGD (100 iters) against **each** of the 5 ground truths, early stop below the score threshold (10 for COCO, 2 for Flickr30k, 0 for VQA), ② single precision APGD, warm started on the ground truth with the lowest score, ③ targeted single precision attacks with `"maybe"` (lower case) and `"Word"` (capitalized, not for TextVQA) from a clean initialization for the VQA tasks; the worst case score per sample is kept |
| stealthy targeted attacks | `attacks/pgd.py` + `eval/lvlm_eval.py` | targeted `l_inf` attack, 10 000 iterations (App. B.9), step size 1/255, on the 6 target captions of App. B.8, 25 images per caption |
| jailbreaking | `eval/jailbreak.py` | universal targeted attack of Qi et al. (2023) adapted to LLaVA-1.5 7B: 5000 iterations, `alpha = 1/255`, no momentum, a single clean image, harmful target strings of the derogatory corpus |

**Precision handling** (`utils/precision.py`) implements the statement of the
addendum that *"for half-precision attacks, 16-bit ints need to be used, and for
single-precision attacks, 32-bit ints need to be used"*: perturbations are
integers in `[-eps_int, eps_int]` (multiples of 1/255) that are snapped onto the
grid of the attack precision, the ball is centred on the clean image *in that
precision*, and every forward pass of the attack runs on the snapped image.

## 4. Reproducing the tables

```bash
# Table 1 — LVLM clean + robust performance (COCO, Flickr30k, TextVQA, VQAv2)
python scripts/eval_lvlm.py --backend llava --task coco \
    --checkpoint checkpoints/FARE4-ViT-L-14-openai.pt --radii 2/255 4/255 \
    --n-adv 500 --output results/llava_coco_fare4.json

# Table 2 — transfer attacks (200 COCO samples)
python scripts/eval_transfer.py --source-backend llava --source-checkpoint none \
    --target-backend llava --target-checkpoint checkpoints/FARE4-ViT-L-14-openai.pt

# Table 3 — stealthy targeted attacks (needs a human/AI judge of the outputs)
python scripts/eval_targeted.py --checkpoint checkpoints/FARE4-ViT-L-14-openai.pt \
    --radii 2/255 4/255 --iterations 10000

# Table 4 — zero-shot classification + AutoAttack
python scripts/eval_zeroshot.py --arch ViT-L-14 --checkpoint checkpoints/FARE4-ViT-L-14-openai.pt \
    --datasets all --radii 2/255 4/255 --output results/zeroshot_fare4.json

# Table 5 / Table 6 — POPE and SQA-I
python scripts/eval_pope_sqa.py --benchmark both \
    --checkpoint checkpoints/FARE4-ViT-L-14-openai.pt

# Table 7 — jailbreaking (attack + human evaluation)
python scripts/eval_jailbreak.py --mode both --eps 64/255 \
    --checkpoint checkpoints/FARE4-ViT-L-14-openai.pt
# then fill the `harmful` column of outputs/jailbreak_eval.csv and aggregate with
# robust_clip.eval.jailbreak.success_rate_from_csv
```

Trends that the reproduction is expected to show (from the paper):

* the original CLIP encoder is **completely non-robust** on all LVLM tasks and in
  zero-shot classification (robust accuracy ≈ 0 at 2/255 and 4/255), while its
  clean performance is the best,
* FARE² keeps the clean performance *close to the original CLIP* and is much more
  robust than the original CLIP, FARE⁴ trades some clean performance for the best
  robustness at 4/255,
* both FARE models dominate the corresponding TeCoA models in clean performance
  (and mostly in robustness too) on the LVLM tasks,
* in zero-shot classification the FARE models have the best robustness/clean
  trade-off on average (Table 4),
* POPE F1: CLIP > FARE > TeCoA, and SQA-I: FARE ≈ CLIP > TeCoA,
* targeted stealth attacks break LLaVA with the original CLIP encoder in 100 % of
  the cases, while TeCoA/FARE are largely robust; the same holds for the
  jailbreaking attempts.

## 5. Repository layout

```
robust_clip/
  utils/         misc helpers, precision aware perturbations, normalization
  models/        OpenCLIP encoder wrapper, zero-shot text classifier
    lvlm/        LLaVA-1.5 (OpenCLIP vision tower) and OpenFlamingo wrappers, prompts
  training/      FARE / TeCoA losses, ImageNet data pipeline, adversarial fine-tuning
  attacks/       PGD, APGD, the LVLM attack ensemble
  eval/          CIDEr, VQA accuracy, zero-shot + AutoAttack, LVLM evaluation,
                 POPE / SQA-I, jailbreaking, datasets
scripts/         command line entry points for every table
tests/           unit and smoke tests (CPU, seconds)
docs/            verification log (what was run, and the cross-checks)
```

Mapping to the paper:

| paper part | code |
| --- | --- |
| Eq. (3), FARE loss | `training/losses.py::fare_loss` |
| Sec. 3.2 / App. B.1, fine-tuning | `training/adv_train.py`, `training/train.py` |
| App. B.1 inner PGD | `attacks/pgd.py` |
| Sec. 4.1 attack setup / App. B.6 | `attacks/lvlm_attack.py` |
| Sec. 4.2 / App. B.8, B.9 targeted attacks | `eval/lvlm_eval.py::targeted_attack(s)` |
| Sec. 4.3 / App. B.10 zero-shot + AutoAttack | `eval/zeroshot.py`, `attacks/apgd.py` |
| Sec. 4.4 jailbreaking (Qi et al.) | `eval/jailbreak.py` |
| Sec. 4.4 POPE / SQA-I | `eval/lvlm_eval.py::evaluate_pope`, `evaluate_sqa`, `eval/datasets.py` |
| Table 1 / Table 2 | `eval/lvlm_eval.py::evaluate_clean`, `evaluate_robust`, `transfer_attack` |
| Table 4 | `eval/zeroshot.py` |
| Table 14 / Eqs. (4), (5) | `training/losses.py`, `eval/lvlm_eval.py::evaluate_embedding_loss` |
| addendum (encoders of LLaVA / OpenFlamingo) | `models/lvlm/llava_openclip.py`, `models/lvlm/open_flamingo.py` |

## 6. Models and datasets

* **LLaVA-1.5 7B** (`liuhaotian/llava-v1.5-7b`) with the **OpenAI CLIP ViT-L/14@224**
  encoder, plugged in through `OpenCLIPVisionTower` (patch tokens of the
  second-to-last layer, `vision_feature_layer = -2`, `select_feature = 'patch'`);
  the projector of the checkpoint and the Vicuna/LLaMA language model stay frozen.
  The prompts are the default LLaVA system prompt plus the task prompts of the
  LLaVA repository (captioning, VQA, POPE, SQA-I).
* **OpenFlamingo-9B** (`openflamingo/OpenFlamingo-9B-vitl-mpt7b`) with the CLIP
  ViT-L/14 encoder exchanged for the robust one; the evaluation uses the
  zero-shot prompts of Alayrac et al./Awadalla et al. (context text, no context
  images).  Requires `pip install git+https://github.com/mlfoundations/open_flamingo.git`.
* **ImageNet-1k** via `datasets.load_dataset("imagenet-1k", trust_remote_code=True)`.
* Captioning: COCO and Flickr30k (5 ground truth captions per image), VQA: VQAv2
  and TextVQA (10 human answers per question, official metric), POPE
  (`coco_pope_{random,popular,adversarial}.json` of the LLaVA repository), SQA-I
  (ScienceQA image questions).
* CIDEr is implemented from scratch (`eval/cider.py`) and validated against the
  reference implementation `pycocoevalcap` — the numerical agreement is exact up
  to the tokenizer, and the reported values use the ×100 scale of the paper.
* the APGD implementation is validated against the reference AutoAttack
  (`fra31/auto-attack`): with **APGD-CE + targeted APGD-DLR (9 target classes,
  100 iterations)** the resulting robust accuracy of the evaluation of Table 4 is
  identical to AutoAttack's on a test network (e.g. 23.4 % vs 23.4 % at
  2/255 and 0 % vs 0 % at 4/255 and 8/255); the initial step size follows the
  paper (`eps`), and the attacks are combined per sample by "misclassified
  wins", which is the worst case for the model.

## 7. What was reproduced, and what is approximated

Reproduced (code complete, unit-tested where possible on CPU):

* the FARE and TeCoA objectives with the exact training configuration of the
  paper, including the class-token-only loss and the two training radii,
* the PGD and APGD implementations with the paper's settings (initial step size
  `eps`, 100 iterations, 16-bit/32-bit integer perturbations, non-normalized
  `l_inf` ball, momentum 0.9 or 0, uniform random initialization),
* the two-stage attack ensemble with the CIDEr/accuracy thresholds and the
  targeted VQA stage (`"Maybe"`, `"Word"`, no `"Word"` for TextVQA),
* the LLaVA wrapper with the OpenCLIP ViT-L/14@224 tower (verified end-to-end with
  a tiny random LLaVA model: image token replacement, gradient flow into the
  perturbed image, generation),
* the OpenFlamingo wrapper with a swappable vision encoder,
* the zero-shot + AutoAttack protocol of App. B.10,
* the targeted stealth attacks and the universal jailbreaking attack,
* POPE / SQA-I evaluation and the transfer-attack harness.

Approximations / deviations that are documented in the code:

* the concrete Hugging Face datasets backing COCO / Flickr30k / VQAv2 / TextVQA /
  SQA-I are configurable (`eval/datasets.py`); the paper uses the original
  dataset releases and 500 randomly sampled images for the adversarial
  evaluations,
* the sixth target caption of Table 3 needs 25 stock photos of patients /
  syringes, which are not redistributable; the harness accepts any 25 images,
* harmfulness in the jailbreaking evaluation is a human decision in the paper:
  the code dumps the model answers to a CSV for labelling
  (`write_human_eval_csv` / `success_rate_from_csv`); the keyword based helper is
  explicitly marked as an approximation,
* the category assignment of the 40 harmful prompts (identity / disinformation /
  crime / x-risk) is not specified in the paper; a keyword based default is
  written to `data/harmful_corpus/categories.json` and can be corrected by hand,
* CIDEr uses a light tokenizer instead of the PTB tokenizer of `pycocoevalcap`,
* SQA-I uses the image subset of ScienceQA; the exact 10k subset of the paper is
  configurable via `--n-samples`.

The offensive content of the jailbreaking experiments (harmful target strings and
prompts) is **not** stored in this repository; it is downloaded at runtime from
the public Qi et al. (2023) repository, exactly as the addendum references it.

## 8. Environment and tests

```bash
pip install -r requirements.txt
python -m pytest tests -q      # 27 unit / smoke tests, CPU only, ~60 s
```

The experiments of the paper were run on GPUs (LLaVA-1.5 7B, OpenFlamingo-9B,
ImageNet adversarial training).  This reproduction environment has no GPU, so the
long runs were not executed here; all scripts are written to run on a GPU machine
(single node or with `--device`), and the code paths that do not need the large
checkpoints are covered by the test suite.
