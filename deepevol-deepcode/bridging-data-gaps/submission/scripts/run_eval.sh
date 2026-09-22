#!/usr/bin/env bash
# =============================================================================
# scripts/run_eval.sh
#
# End-to-end evaluation driver for DPMs-ANT (paper Sections 5.2 / 5.3).
#
# Pipeline per (source -> target) task and per backbone (ddpm | ldm):
#   1. (optional) fine-tune the 2-way source/target classifier  p_phi
#        -> scripts/train_classifier.py
#   2. (optional) train the zero-initialized adaptors  psi  with
#      adversarial-noise + similarity-guided objective (Algorithm 1, Eq. 7/8)
#        -> scripts/train_ant.py
#   3. sample 1,000 images with the adapted reverse process
#      (DDIM eta=0, 100 steps by default; DDPM eta=1, T=1000 optional)
#        -> scripts/sample.py
#   4. score the generated images with Intra-LPIPS (higher is better, §5.2)
#      and FID against the larger target sets
#      (Sunglasses 2.5k / Babies 2.7k, lower is better)
#        -> python -m dpm_ant.evaluation.evaluate
#
# Everything is graceful: a step is skipped when its inputs are absent, so the
# script can be used to (a) reproduce the paper tables end-to-end or
# (b) only evaluate pre-generated images from baselines (see baselines/).
#
# Usage:
#   bash scripts/run_eval.sh                         # all tasks, ddpm+ldm
#   BACKBONES=ddpm bash scripts/run_eval.sh          # ddpm only
#   TASKS="ffhq_sunglasses" bash scripts/run_eval.sh
#   SKIP_TRAIN=1 SKIP_CLASSIFIER=1 bash scripts/run_eval.sh
#   NUM_SAMPLES=200 NUM_STEPS=50 SMOKE=1 bash scripts/run_eval.sh
#
# Environment variables (all optional):
#   TASKS            space separated task keys (default: full registry below)
#   BACKBONES        "ddpm ldm" (default: both)
#   CONFIG           path to configs/default.yaml
#   PER_TASK_CONFIG  path to configs/per_task.yaml
#   CLASSIFIER_CONFIG path to configs/classifier.yaml
#   OUT_ROOT         root for checkpoints/samples/reports
#   NUM_SAMPLES      images generated per task (paper: 1000)
#   NUM_STEPS        reverse-process steps (paper: DDIM 100 for evaluation)
#   ETA              sampler eta (0 = DDIM, 1 = DDPM)
#   METHOD           "ddim" or "ddpm"
#   SEEDS            comma separated seeds (default: 0)
#   GPU              CUDA_VISIBLE_DEVICES value (default: 0)
#   SKIP_CLASSIFIER  1 -> never run the classifier fine-tuning step
#   SKIP_TRAIN       1 -> reuse existing adaptor checkpoints
#   SKIP_SAMPLE      1 -> reuse existing generated images
#   SKIP_FID         1 -> compute Intra-LPIPS only
#   SKIP_INTRA_LPIPS 1 -> compute FID only
#   FID_BACKEND      clean-fid | pytorch-fid | torch
#   SMOKE            1 -> tiny run (50 samples / 10 steps / 5 iterations)
#   DRY_RUN          1 -> print the commands without executing them
# =============================================================================

set -u -o pipefail

# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG="${CONFIG:-configs/default.yaml}"
PER_TASK_CONFIG="${PER_TASK_CONFIG:-configs/per_task.yaml}"
CLASSIFIER_CONFIG="${CLASSIFIER_CONFIG:-configs/classifier.yaml}"

OUT_ROOT="${OUT_ROOT:-outputs}"
LOG_DIR="${OUT_ROOT}/logs"
CKPT_DIR="${OUT_ROOT}/checkpoints"
SAMPLE_ROOT="${OUT_ROOT}/samples"
REPORT_ROOT="${OUT_ROOT}/reports"
CLASSIFIER_CKPT_DIR="${OUT_ROOT}/classifier"

# --------------------------------------------------------------------------- #
# Paper hyperparameters / defaults (Section 5.2, Table 1)
# --------------------------------------------------------------------------- #
NUM_SAMPLES="${NUM_SAMPLES:-1000}"      # §5.2: generate 1,000 images for Intra-LPIPS
NUM_STEPS="${NUM_STEPS:-100}"           # §5.2: DDIM 100 steps for evaluation
METHOD="${METHOD:-ddim}"                # "ddim" (eta=0) or "ddpm" (eta=1, T=1000)
ETA="${ETA:-0.0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
FID_BACKEND="${FID_BACKEND:-clean-fid}"
CLASSIFIER_ITERS="${CLASSIFIER_ITERS:-300}"   # addendum: Adam, lr 1e-4, bs 64, 300 iters
ANT_ITERS="${ANT_ITERS:-300}"                 # §5.2: ~300 outer iterations

SEEDS="${SEEDS:-0}"
BACKBONES="${BACKBONES:-ddpm ldm}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"

SKIP_CLASSIFIER="${SKIP_CLASSIFIER:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_SAMPLE="${SKIP_SAMPLE:-0}"
SKIP_FID="${SKIP_FID:-0}"
SKIP_INTRA_LPIPS="${SKIP_INTRA_LPIPS:-0}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"

if [ "${SMOKE}" = "1" ]; then
  NUM_SAMPLES="${NUM_SAMPLES:-50}"
  NUM_STEPS="${NUM_STEPS:-10}"
  ANT_ITERS="${ANT_ITERS:-5}"
fi

# --------------------------------------------------------------------------- #
# Task registry (source -> 10-shot target), mirrors configs/per_task.yaml
# --------------------------------------------------------------------------- #
DEFAULT_TASKS="ffhq_babies ffhq_sunglasses ffhq_raphael church_haunted_houses church_landscape_drawings"
TASKS="${TASKS:-${DEFAULT_TASKS}}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${SAMPLE_ROOT}" "${REPORT_ROOT}" "${CLASSIFIER_CKPT_DIR}"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
log()  { printf '\033[1;34m[run_eval]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[run_eval][warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[run_eval][error]\033[0m %s\n' "$*" >&2; exit 1; }

run() {
  # Execute a command, honouring DRY_RUN, logging stdout/stderr to a file.
  local logfile="$1"; shift
  if [ "${DRY_RUN}" = "1" ]; then
    printf '  [dry-run] %s\n' "$*"
    return 0
  fi
  printf '  $ %s\n' "$*"
  "$@" >>"${logfile}" 2>&1
  return $?
}

task_target() {
  # Strip the backbone suffix (_ldm) and the source prefix (ffhq_/church_).
  local t="$1"
  t="${t%_ldm}"
  t="${t#ffhq_}"
  t="${t#church_}"
  printf '%s' "${t}"
}

task_backbone() {
  # Explicit `_ldm` suffix wins, otherwise use the caller-provided backbone.
  local t="$1" default_bb="$2"
  case "${t}" in
    *_ldm) printf 'ldm' ;;
    *)     printf '%s' "${default_bb}" ;;
  esac
}

ckpt_path() { printf '%s/%s_%s_ant.pt' "${CKPT_DIR}" "$1" "$2"; }
sample_dir() { printf '%s/%s_%s' "${SAMPLE_ROOT}" "$1" "$2"; }

# --------------------------------------------------------------------------- #
# Consistency check: the requested backbone must be the task's backbone
# --------------------------------------------------------------------------- #
for TASK in ${TASKS}; do
  for BB in ${BACKBONES}; do
    ACTUAL_BB="$(task_backbone "${TASK}" "${BB}")"
    if [ "${ACTUAL_BB}" != "${BB}" ]; then
      warn "task '${TASK}' is an LDM task; skipping the ddpm pass"
      continue
    fi
    :
  done
done

# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
log "repo root        : ${REPO_ROOT}"
log "config           : ${CONFIG} (+ ${PER_TASK_CONFIG})"
log "tasks            : ${TASKS}"
log "backbones        : ${BACKBONES}"
log "samples/steps    : ${NUM_SAMPLES} / ${NUM_STEPS} (${METHOD}, eta=${ETA})"
log "outputs          : ${OUT_ROOT}"
log "python           : ${PYTHON_BIN}"

TOTAL_TASKS=0
FAILED_TASKS=0

for TASK in ${TASKS}; do
  TARGET="$(task_target "${TASK}")"

  for BB in ${BACKBONES}; do
    ACTUAL_BB="$(task_backbone "${TASK}" "${BB}")"
    if [ "${ACTUAL_BB}" != "${BB}" ]; then
      continue
    fi
    TOTAL_TASKS=$((TOTAL_TASKS + 1))

    RUN_TAG="${TASK}_${BB}"
    LOGFILE="${LOG_DIR}/${RUN_TAG}.log"
    TASK_SAMPLE_DIR="$(sample_dir "${TASK}" "${BB}")"
    TASK_CKPT="$(ckpt_path "${TASK}" "${BB}")"
    REPORT="${REPORT_ROOT}/${RUN_TAG}.json"
    CLS_CKPT="${CLASSIFIER_CKPT_DIR}/classifier_${TASK}_${BB}.pt"
    ADAPTOR_ONLY=""

    if [ "${BB}" = "ldm" ]; then
      ADAPTOR_ONLY="--adaptor-only"
    fi

    log "=================================================================="
    log "task ${RUN_TAG}  (target=${TARGET}, backbone=${BB})"
    log "=================================================================="

    # ---------------------------------------------------------------- #
    # Step 1 - classifier fine-tuning (Section 4.1 / 5.2, addendum)
    #          Adam, lr=1e-4, batch 64, 300 iterations on noised
    #          source+target images with t ~ Uniform({1..T}).
    # ---------------------------------------------------------------- #
    if [ "${SKIP_CLASSIFIER}" = "1" ] || [ -f "${CLS_CKPT}" ]; then
      log "  [1/4] classifier: skipping (found ${CLS_CKPT} or SKIP_CLASSIFIER=1)"
    else
      log "  [1/4] classifier fine-tuning -> ${CLS_CKPT}"
      run "${LOGFILE}" "${PYTHON_BIN}" scripts/train_classifier.py \
        --config "${CONFIG}" \
        --per-task-config "${PER_TASK_CONFIG}" \
        --classifier-config "${CLASSIFIER_CONFIG}" \
        --task "${TASK}" \
        --backbone "${BB}" \
        --target "${TARGET}" \
        --iterations "${CLASSIFIER_ITERS}" \
        --out "${CLASSIFIER_CKPT_DIR}" \
        --seed 0 || warn "classifier step failed for ${RUN_TAG} (see ${LOGFILE})"
    fi

    # ---------------------------------------------------------------- #
    # Step 2 - ANT adaptor training (Algorithm 1, Eq. 7 inner ascent and
    #          Eq. 8 outer similarity-guided objective). Only psi is updated:
    #          theta (frozen backbone) and phi (frozen classifier).
    #          Ablation: add --no-adv-noise for "DPMs-ANT w/o AN",
    #                    add --full-finetune for direct full-model tuning.
    # ---------------------------------------------------------------- #
    if [ "${SKIP_TRAIN}" = "1" ] || [ -f "${TASK_CKPT}" ]; then
      log "  [2/4] ANT training: skipping (found ${TASK_CKPT} or SKIP_TRAIN=1)"
    else
      log "  [2/4] ANT adaptor training -> ${TASK_CKPT}"
      # shellcheck disable=SC2086
      run "${LOGFILE}" "${PYTHON_BIN}" scripts/train_ant.py \
        --config "${CONFIG}" \
        --per-task-config "${PER_TASK_CONFIG}" \
        --task "${TASK}" \
        --backbone "${BB}" \
        --target "${TARGET}" \
        --classifier-checkpoint "${CLS_CKPT}" \
        --iterations "${ANT_ITERS}" \
        --gamma 5.0 \
        --omega 0.02 \
        --J 10 \
        --norm per_sample \
        --out "${CKPT_DIR}" \
        --seed 0 ${ADAPTOR_ONLY} || warn "ANT training failed for ${RUN_TAG} (see ${LOGFILE})"
    fi

    # ---------------------------------------------------------------- #
    # Step 3 - sampling with the adapted reverse process (Eq. 2 / Eq. 3),
    #          DDIM eta=0 with `NUM_STEPS` steps by default, optionally
    #          guided by the frozen classifier (Eq. 4).
    # ---------------------------------------------------------------- #
    if [ "${SKIP_SAMPLE}" = "1" ] || [ -d "${TASK_SAMPLE_DIR}" ]; then
      log "  [3/4] sampling: skipping (found ${TASK_SAMPLE_DIR} or SKIP_SAMPLE=1)"
    else
      log "  [3/4] sampling ${NUM_SAMPLES} images -> ${TASK_SAMPLE_DIR}"
      run "${LOGFILE}" "${PYTHON_BIN}" scripts/sample.py \
        --config "${CONFIG}" \
        --per-task-config "${PER_TASK_CONFIG}" \
        --task "${TASK}" \
        --backbone "${BB}" \
        --method "${METHOD}" \
        --num-samples "${NUM_SAMPLES}" \
        --num-steps "${NUM_STEPS}" \
        --eta "${ETA}" \
        --batch-size "${BATCH_SIZE}" \
        --adaptor-checkpoint "${TASK_CKPT}" \
        --classifier-checkpoint "${CLS_CKPT}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --out "${TASK_SAMPLE_DIR}" \
        --seed 0 || { warn "sampling failed for ${RUN_TAG}"; FAILED_TASKS=$((FAILED_TASKS + 1)); continue; }
    fi

    # ---------------------------------------------------------------- #
    # Step 4 - evaluation: Intra-LPIPS (higher is better, §5.2) and FID
    #          against the larger target sets (Babies 2.7k / Sunglasses 2.5k,
    #          lower is better). Both metrics are computed on the same
    #          generated images.
    # ---------------------------------------------------------------- #
    EVAL_FLAGS=""
    if [ "${SKIP_INTRA_LPIPS}" = "1" ]; then EVAL_FLAGS="${EVAL_FLAGS} --no-intra-lpips"; fi
    if [ "${SKIP_FID}" = "1" ]; then EVAL_FLAGS="${EVAL_FLAGS} --no-fid"; fi

    log "  [4/4] evaluation -> ${REPORT}"
    # shellcheck disable=SC2086
    if run "${LOGFILE}" "${PYTHON_BIN}" -m dpm_ant.evaluation.evaluate \
        --config "${CONFIG}" \
        --task "${TASK}" \
        --backbone "${BB}" \
        --generated-dir "${TASK_SAMPLE_DIR}" \
        --reference-dir "${TARGET:-}" \
        --split "${TARGET}" \
        --num-samples "${NUM_SAMPLES}" \
        --num-steps "${NUM_STEPS}" \
        --fid-backend "${FID_BACKEND}" \
        --report "${REPORT}" \
        --seed "${SEEDS}" ${EVAL_FLAGS}; then
      log "  done -> ${REPORT}"
    else
      warn "evaluation failed for ${RUN_TAG} (see ${LOGFILE})"
      FAILED_TASKS=$((FAILED_TASKS + 1))
    fi
  done
done

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
log "=================================================================="
log "run_eval finished: ${TOTAL_TASKS} task/backbone passes, ${FAILED_TASKS} failure(s)"
log "samples   : ${SAMPLE_ROOT}"
log "reports   : ${REPORT_ROOT}"
log "logs      : ${LOG_DIR}"
log "reference paper numbers (Tables 1-2, 4):"
log "  Intra-LPIPS  LSUN Church -> Landscape drawings : DDPM-PA 0.706 | DDPM-ANT 0.723 | LDM-ANT 0.738"
log "  FID          FFHQ -> Babies                    : 46.70"
log "  FID          FFHQ -> Sunglasses                : 20.06"
log "  parameter rate (fine-tuned / total)            : DDPM-ANT 1.3% | LDM-ANT 1.6%"
log "=================================================================="

log "To summarise all JSON reports into a single markdown/CSV table run:"
log "  python - <<'PY'"
log "  import glob, json; [print(p, json.load(open(p)).get('method'), json.load(open(p)).get('intra_lpips'), json.load(open(p)).get('fid')) for p in sorted(glob.glob('${REPORT_ROOT}/*.json'))]"
log "  PY"

exit 0
