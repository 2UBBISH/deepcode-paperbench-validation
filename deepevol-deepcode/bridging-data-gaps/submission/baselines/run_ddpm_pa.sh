#!/usr/bin/env bash
# =============================================================================
# baselines/run_ddpm_pa.sh
# -----------------------------------------------------------------------------
# Driver for the DDPM-PA baseline (Zhu et al. 2022), which is one of the main
# comparison methods in DPMs-ANT (Tables 1-2, Appendix B.2 Table 4,
# Appendix B.4 Table 9).
#
# DDPM-PA ("DDPM Patch-based Adaptation", *Diverse Image Generation from
# Few Samples using Diffusion Models*) fine-tunes a pretrained DDPM at a lower
# resolution and then patches the low-resolution weights into the high-resolution
# U-Net; it is evaluated here with EXACTLY the same Intra-LPIPS / FID scripts as
# DPMs-ANT, so the numbers are directly comparable (apples-to-apples).
#
# This script does NOT reimplement DDPM-PA. It:
#   1. renders the official DDPM-PA train / sample commands (command templates
#      live in `baselines/eval_baselines.py` -> `BASELINES["ddpm_pa"]`),
#   2. evaluates the resulting generated images with the shared DPMs-ANT metric
#      implementation (`dpm_ant.evaluation`), and
#   3. diffs the measured numbers against the paper's reported values.
#
# Usage:
#   bash baselines/run_ddpm_pa.sh                # full sweep, env-var driven
#   DRY_RUN=1 bash baselines/run_ddpm_pa.sh      # print commands only
#   TASKS="ffhq_sunglasses" EVAL_ONLY=1 bash baselines/run_ddpm_pa.sh
#
# All configuration is via environment variables (see the block below).
# =============================================================================

set -u -o pipefail

# -----------------------------------------------------------------------------
# Repo root / interpreter
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
DRIVER="${SCRIPT_DIR}/eval_baselines.py"

# -----------------------------------------------------------------------------
# Configuration (env-var overridable)
# -----------------------------------------------------------------------------
CONFIG="${CONFIG:-configs/default.yaml}"
PER_TASK_CONFIG="${PER_TASK_CONFIG:-configs/per_task.yaml}"
CLASSIFIER_CONFIG="${CLASSIFIER_CONFIG:-configs/classifier.yaml}"

# Baseline method registry key inside baselines/eval_baselines.py BASELINES.
METHOD="${METHOD:-ddpm_pa}"

# Source -> target tasks. Task keys match configs/per_task.yaml and
# dpm_ant/data/datasets.py registries. Only tasks whose source backbone is a
# pretrained DDPM are meaningful for DDPM-PA; LDM tasks are automatically
# skipped by the driver unless ALLOW_LDM=1.
TASKS="${TASKS:-ffhq_babies ffhq_sunglasses ffhq_raphael church_haunted_houses church_landscape_drawings}"
ALLOW_LDM="${ALLOW_LDM:-0}"

# Where the official baseline codebase is checked out (see baselines/README.md)
# and where the baseline is expected to write its generated images.
EXTERNAL_ROOT="${EXTERNAL_ROOT:-external}"
OUT_ROOT="${OUT_ROOT:-outputs/baselines}"

# Data layout (must match dpm_ant/data/datasets.py + configs/default.yaml)
SOURCE_DIR="${SOURCE_DIR:-}"
TARGET_DIR="${TARGET_DIR:-}"
FID_TARGET_DIR="${FID_TARGET_DIR:-}"

# Dataset / evaluation settings (paper Section 5.2)
SHOTS="${SHOTS:-10}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-2500}"
FID_BACKEND="${FID_BACKEND:-clean-fid}"
DEVICE="${DEVICE:-}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
SEED="${SEED:-0}"

# Skip flags / modes
EVAL_ONLY="${EVAL_ONLY:-0}"     # skip training, only sample (or only evaluate)
SKIP_TRAIN="${SKIP_TRAIN:-0}"   # never run the official training command
SKIP_SAMPLE="${SKIP_SAMPLE:-0}" # reuse already-generated images
SKIP_EVAL="${SKIP_EVAL:-0}"     # only train/sample
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
VERBOSE="${VERBOSE:-1}"
STRICT="${STRICT:-0}"           # 1 -> abort the task on failure

# Smoke-test overrides: tiny sample counts, one task, one seed
if [[ "${SMOKE}" == "1" ]]; then
  TASKS="ffhq_sunglasses"
  NUM_SAMPLES="${SMOKE_NUM_SAMPLES:-8}"
  FID_NUM_SAMPLES="${SMOKE_FID_NUM_SAMPLES:-0}"
  SKIP_EVAL="${SMOKE_SKIP_EVAL:-0}"
fi

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}" 2>/dev/null || true
LOG_FILE="${LOG_DIR}/run_ddpm_pa.log"

_ts() { date "+%Y-%m-%d %H:%M:%S"; }

log() {
  local msg="$*"
  printf "[%s] %s\n" "$(_ts)" "${msg}"
  printf "[%s] %s\n" "$(_ts)" "${msg}" >>"${LOG_FILE}" 2>/dev/null || true
}

warn() {
  printf "[%s] WARNING: %s\n" "$(_ts)" "$*" >&2
  printf "[%s] WARNING: %s\n" "$(_ts)" "$*" >>"${LOG_FILE}" 2>/dev/null || true
}

die() {
  printf "[%s] ERROR: %s\n" "$(_ts)" "$*" >&2
  printf "[%s] ERROR: %s\n" "$(_ts)" "$*" >>"${LOG_FILE}" 2>/dev/null || true
  exit 1
}

# run <description> -- <cmd...>
# Executes a command (or prints it when DRY_RUN=1) and appends output to a log.
run() {
  local desc="$1"; shift
  if [[ "${1:-}" == "--" ]]; then shift; fi
  log "-> ${desc}"
  log "   cmd: $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@" >>"${LOG_FILE}" 2>&1
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    warn "command failed (rc=${rc}): $*"
  fi
  return ${rc}
}

# -----------------------------------------------------------------------------
# Task helpers
# -----------------------------------------------------------------------------
task_backbone() {
  # Mirror dpm_ant/evaluation/evaluate.py + scripts/train_ant.py heuristics.
  local task="$1"
  case "${task}" in
    *_ldm) printf "ldm\n" ;;
    *_ddpm) printf "ddpm\n" ;;
    *) printf "ddpm\n" ;;
  esac
}

task_target() {
  # ffhq_babies -> babies ; church_landscape_drawings -> landscape_drawings
  local task="$1"
  task="${task%_ldm}"; task="${task%_ddpm}"
  printf "%s\n" "${task#*_}"
}

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
if [[ ! -f "${DRIVER}" ]]; then
  die "baseline driver not found: ${DRIVER}"
fi
if [[ ! -f "${CONFIG}" ]]; then
  warn "config not found: ${CONFIG} (the driver will use its built-in defaults)"
fi

log "=============================================================="
log "DDPM-PA baseline evaluation (Zhu et al. 2022)"
log "method=${METHOD}  interpreter=${PYTHON_BIN}"
log "tasks=${TASKS}"
log "out_root=${OUT_ROOT}  external_root=${EXTERNAL_ROOT}"
log "shots=${SHOTS}  image_size=${IMAGE_SIZE}  num_samples=${NUM_SAMPLES}"
log "eval_only=${EVAL_ONLY} skip_train=${SKIP_TRAIN} skip_sample=${SKIP_SAMPLE} skip_eval=${SKIP_EVAL}"
log "dry_run=${DRY_RUN} smoke=${SMOKE}"
log "=============================================================="

# Show the registry once (helps users see which baselines are runnable).
if [[ "${VERBOSE}" == "1" ]]; then
  run "list baseline registry" -- \
    "${PYTHON_BIN}" "${DRIVER}" --list
fi

# -----------------------------------------------------------------------------
# Main loop over tasks
# -----------------------------------------------------------------------------
FAILED_TASKS=""
RUN_TASKS=0

for TASK in ${TASKS}; do
  BACKBONE="$(task_backbone "${TASK}")"
  TARGET="$(task_target "${TASK}")"

  if [[ "${BACKBONE}" == "ldm" && "${ALLOW_LDM}" != "1" ]]; then
    log "skip ${TASK}: DDPM-PA is a pixel-space DDPM baseline (set ALLOW_LDM=1 to force)"
    continue
  fi

  RUN_TASKS=$((RUN_TASKS + 1))
  log "--------------------------------------------------------------"
  log "task=${TASK}  backbone=${BACKBONE}  target=${TARGET}"
  log "--------------------------------------------------------------"

  # Common CLI arguments accepted by baselines/eval_baselines.py.
  CMD=(
    "${PYTHON_BIN}" "${DRIVER}"
    --method "${METHOD}"
    --task "${TASK}"
    --backbone "${BACKBONE}"
    --target "${TARGET}"
    --out-root "${OUT_ROOT}"
    --external-root "${EXTERNAL_ROOT}"
    --config "${CONFIG}"
    --per-task-config "${PER_TASK_CONFIG}"
    --classifier-config "${CLASSIFIER_CONFIG}"
    --shots "${SHOTS}"
    --image-size "${IMAGE_SIZE}"
    --num-samples "${NUM_SAMPLES}"
    --fid-num-samples "${FID_NUM_SAMPLES}"
    --fid-backend "${FID_BACKEND}"
    --seed "${SEED}"
    --gpu "${GPU}"
    --python "${PYTHON_BIN}"
  )

  [[ -n "${SOURCE_DIR}" ]]     && CMD+=(--source-dir "${SOURCE_DIR}")
  [[ -n "${TARGET_DIR}" ]]     && CMD+=(--target-dir "${TARGET_DIR}")
  [[ -n "${FID_TARGET_DIR}" ]] && CMD+=(--fid-target-dir "${FID_TARGET_DIR}")
  [[ -n "${DEVICE}" ]]         && CMD+=(--device "${DEVICE}")

  [[ "${DRY_RUN}" == "1" ]]    && CMD+=(--dry-run)
  [[ "${VERBOSE}" == "1" ]]    && CMD+=(--verbose)
  [[ "${EVAL_ONLY}" == "1" ]]  && CMD+=(--eval-only)
  [[ "${SKIP_TRAIN}" == "1" ]] && CMD+=(--skip-train)
  [[ "${SKIP_SAMPLE}" == "1" ]]&& CMD+=(--skip-sample)
  [[ "${SKIP_EVAL}" == "1" ]]  && CMD+=(--skip-eval)

  REPORT_PATH="${OUT_ROOT}/${METHOD}/${TASK}_${BACKBONE}/report.json"
  CMD+=(--report "${REPORT_PATH}")

  if run "DDPM-PA ${TASK} (train -> sample -> Intra-LPIPS/FID)" -- "${CMD[@]}"; then
    log "ok: ${TASK}"
  else
    warn "failed: ${TASK}"
    FAILED_TASKS="${FAILED_TASKS} ${TASK}"
    if [[ "${STRICT}" == "1" ]]; then
      die "aborting (STRICT=1) after failure on ${TASK}"
    fi
  fi
done

if [[ "${RUN_TASKS}" -eq 0 ]]; then
  warn "no tasks were executed"
fi

# -----------------------------------------------------------------------------
# Paper reference tables (Tables 1-2, 4, 9) for manual comparison
# -----------------------------------------------------------------------------
log "paper reference numbers (DDPM-PA column of Tables 1-2, 4, 9):"
run "print paper tables" -- "${PYTHON_BIN}" "${DRIVER}" --paper

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
log "=============================================================="
log "reports written under: ${OUT_ROOT}/${METHOD}/"
log "per-task logs:         ${LOG_FILE}"
if [[ -n "${FAILED_TASKS}" ]]; then
  warn "tasks with failures:${FAILED_TASKS}"
  log "=============================================================="
  exit 1
fi
log "all DDPM-PA baseline runs completed"
log "note: DDPM-PA is trained for 5,000 iterations and needs ~4.2 GPU hours,"
log "      vs. DPMs-ANT which fine-tunes only 1.3% of parameters for ~300"
log "      iterations in ~3 GPU hours (Section 5.3 / Table 8)."
log "=============================================================="
exit 0
