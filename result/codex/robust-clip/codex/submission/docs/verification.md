# Verification log

This file records what was actually executed while building the reproduction and
what the results were.  The environment has **no GPU** and the paper's
experiments (ImageNet adversarial fine-tuning, LLaVA-1.5 7B / OpenFlamingo-9B
evaluations) are long-running, so those runs are *not* executed here; the code is
written for a GPU machine (see `run_all.sh`).

## Automated tests

`python -m pytest tests -q` → **27 passed** (CPU, ~1 min):

| test file | covers |
| --- | --- |
| `tests/test_losses.py` | FARE loss (Eq. (3)) incl. the `lambda` weight and the ℓ1 ablation of App. B.4, no-gradient reference encoder, TeCoA loss, embedding losses of Eqs. (4)/(5) |
| `tests/test_attacks.py` | `eps = k/255` parsing, the precision grid (16-bit snapping, ball around the non-normalized input), APGD-CE and APGD-targeted-DLR reducing accuracy and respecting the ball, DLR loss ordering, PGD without momentum (jailbreaking setting), the "misclassified wins" combination |
| `tests/test_metrics.py` | CIDEr ordering (exact > paraphrase > unrelated), CIDEr batch API, tokenizer, the official VQA accuracy, answer normalization, most frequent answers, POPE F1 and parsing, SQA answer parsing |
| `tests/test_pipeline_smoke.py` | a FARE training step reduces the loss; the ensemble attack produces per-sample worst cases; one real FARE training step on a randomly initialized ViT-B/32 (only the vision tower receives gradients) |
| `tests/test_llava_wrapper.py` | `tokenizer_image_token` (`<image>` → −200), replacement of the image token by the projected 49 OpenCLIP patch tokens, gradient flow from the loss into the *image* (the property the attacks rely on), generation, and the full half/single precision ensemble attack on a real (tiny) LLaVA model |

## Cross-checks against reference implementations

* **CIDEr** (`robust_clip/eval/cider.py`) vs `pycocoevalcap` (the implementation
  used by the COCO challenge): identical scores to machine precision on 80 random
  images with identical tokenization, for exact matches, random captions and
  generic captions (max difference `0.0`).  The ×100 scaling used in the paper's
  tables is `Cider.scale = 100`.
* **APGD** (`robust_clip/attacks/apgd.py`) vs the reference AutoAttack
  (`fra31/auto-attack`, the implementation referenced by the addendum for the
  APGD algorithm).  With APGD-CE + targeted APGD-DLR (9 target classes, 100
  iterations, ℓ∞) the robust accuracy is identical on a test network:

  | eps | AutoAttack (reference) | this reproduction |
  | --- | --- | --- |
  | 2/255 | 23.4 % | 23.4 % |
  | 4/255 | 0.0 % | 0.0 % |
  | 8/255 | 0.0 % | 0.0 % |

  Per-sample cross entropy losses of APGD-CE agree up to ~0.02 (numerical noise
  of the step-size controller), and the number of misclassified samples is
  identical.  The initial step size follows the paper (`eps`), the reference
  default is `2*eps`; both give the same result on this test network.

## Manual checks

* `python -c "import robust_clip"` and `--help` of all nine scripts in `scripts/`
  run (CLI wiring, imports and argument parsing).
* A real (randomly initialized) OpenCLIP `ViT-B-32` was used to check the vision
  feature extraction: 49 patch tokens for 224/32 input, class token, second-to-last
  layer selection, projected embedding and a complete APGD run over the vision
  tower + zero-shot classifier head.
* The jailbreaking corpus is downloaded from the public Qi et al. (2023)
  repository at runtime (the URLs of the addendum) — the content is intentionally
  not stored in this repository.

## What could not be verified here

* the numerical results of Tables 1–7 (they need the fine-tuned ViT-L/14
  encoders, ImageNet, LLaVA-1.5 7B and OpenFlamingo-9B on GPUs),
* OpenFlamingo inference (the optional `open_flamingo` package and the 9B
  checkpoint are not installed here),
* the half precision stage of the ensemble attack on the real LVLMs: torch does
  not implement several fp16 kernels on CPU, so the code falls back to fp32 on
  CPU with a warning (on GPU the first stage runs in genuine half precision).
