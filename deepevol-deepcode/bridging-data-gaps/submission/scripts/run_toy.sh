#!/usr/bin/env bash
# =============================================================================
# scripts/run_toy.sh
# -----------------------------------------------------------------------------
# Driver for the Section 5.1 toy 2-D Gaussian experiment reproducing Figure 2
# of "Adapting Pretrained Diffusion Models for Few-Shot Image Generation"
# (DPMs-ANT).
#
# The toy experiment is the cheapest end-to-end sanity check of the method:
#   * source  ~ N(mean=(1, 1),  I)
#   * target  ~ N(mean=(-1, -1), I)
#   * a small MLP is pretrained on the source, a binary source/target
#     classifier is trained, and ANT transfers the model to the target using
#     only 10 target samples (adversarial noise selection, Eq. 7, plus the
#     similarity-guided loss, Eq. 5/8).
#
# Figure 2 panels produced:
#   (a) gradient-direction comparison + adversarial-noise cloud / ellipse
#   (b) heat-map baseline   (x = diffusion timestep, y = sampled value)
#   (c) heat-map ANT        (x = diffusion timestep, y = sampled value)
#
# Everything below is overridable through environment variables so the script
# can be used for smoke tests in CI as well as the full reproduction.
#
# Usage:
#   bash scripts/run_toy.sh
#   SMOKE=1 bash scripts/run_toy.sh
#   GAMMA=5.0 OMEGA=0.02 J=10 ITERATIONS=300 SEEDS="0 1 2" bash scripts/run_toy.sh
# =============================================================================

set -u -o pipefail

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

# -----------------------------------------------------------------------------
# Configuration (environment variables, all optional)
# -----------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/default.yaml}"
PER_TASK_CONFIG="${PER_TASK_CONFIG:-configs/per_task.yaml}"
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/toy}"

# Toy / ANT hyperparameters (paper Section 5.1 defaults).
GAMMA="${GAMMA:-5.0}"                 # similarity-guided weight (Eq. 5/8)
OMEGA="${OMEGA:-0.02}"                # adversarial noise step size (Eq. 7)
J="${J:-10}"                          # inner maximization steps (Eq. 7)
ITERATIONS="${ITERATIONS:-300}"       # ANT outer iterations
BATCH_SIZE="${BATCH_SIZE:-10}"        # 10-shot target batch
LR="${LR:-5e-5}"                      # adaptor learning rate
NUM_TIMESTEPS="${NUM_TIMESTEPS:-1000}"
SCHEDULE="${SCHEDULE:-linear}"
NORM="${NORM:-per_sample}"
REFERENCE_SAMPLES="${REFERENCE_SAMPLES:-10000}"   # Fig 2(a) reference gradient
HEATMAP_SAMPLES="${HEATMAP_SAMPLES:-20000}"       # Fig 2(b)/(c) samples
SEEDS="${SEEDS:-0}"

# Source / target Gaussians.
SOURCE_MEAN="${SOURCE_MEAN:-1.0,1.0}"
TARGET_MEAN="${TARGET_MEAN:--1.0,-1.0}"

# Execution flags.
DEVICE="${DEVICE:-}"            # e.g. "cuda" / "cpu"; empty => auto
PLOT="${PLOT:-1}"               # 1 => render Figure 2, 0 => results only
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
VERBOSE="${VERBOSE:-1}"

# -----------------------------------------------------------------------------
# Smoke-test overrides: fast, tiny run that still exercises the full pipeline.
# -----------------------------------------------------------------------------
if [[ "${SMOKE}" == "1" ]]; then
  ITERATIONS="${SMOKE_ITERATIONS:-20}"
  REFERENCE_SAMPLES="${SMOKE_REFERENCE_SAMPLES:-500}"
  HEATMAP_SAMPLES="${SMOKE_HEATMAP_SAMPLES:-200}"
  NUM_TIMESTEPS="${SMOKE_NUM_TIMESTEPS:-100}"
  SEEDS="${SMOKE_SEEDS:-0}"
  OUT_DIR="${SMOKE_OUT_DIR:-${OUT_ROOT}/toy_smoke}"
fi

mkdir -p "${OUT_DIR}/logs" || true
LOG_FILE="${OUT_DIR}/logs/run_toy.log"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "${msg}"
  echo "${msg}" >> "${LOG_FILE}" 2>/dev/null || true
}

warn() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $*"
  echo "${msg}" >&2
  echo "${msg}" >> "${LOG_FILE}" 2>/dev/null || true
}

die() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*"
  echo "${msg}" >&2
  echo "${msg}" >> "${LOG_FILE}" 2>/dev/null || true
  exit 1
}

# run <description> -- <command...>
run() {
  local desc="$1"
  shift
  [[ "${1:-}" == "--" ]] && shift
  log "RUN: ${desc}"
  log "CMD: $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "  (dry-run: command not executed)"
    return 0
  fi
  if "$@" >> "${LOG_FILE}" 2>&1; then
    log "OK : ${desc}"
    return 0
  fi
  local rc=$?
  warn "FAILED (rc=${rc}): ${desc}"
  return ${rc}
}

# Expand a comma-separated "a,b" into a list of individual values.
split_commas() {
  local raw="$1"
  local out=""
  local IFS=','
  for tok in ${raw}; do
    out="${out} ${tok}"
  done
  echo "${out}"
}

SOURCE_MEAN_LIST="$(split_commas "${SOURCE_MEAN}")"
TARGET_MEAN_LIST="$(split_commas "${TARGET_MEAN}")"

# Optional device flag.
DEVICE_ARGS=()
if [[ -n "${DEVICE}" ]]; then
  DEVICE_ARGS=(--device "${DEVICE}")
fi

# Optional plot flag handled by passing the output dir to the plots CLI.
# -----------------------------------------------------------------------------
# Sanity check
# -----------------------------------------------------------------------------
if [[ ! -f "dpm_ant/toy/toy_2d.py" ]]; then
  die "dpm_ant/toy/toy_2d.py not found (run from the repository root)."
fi

log "======================================================================"
log "DPMs-ANT toy 2-D Gaussian experiment (Section 5.1 / Figure 2)"
log "======================================================================"
log "repo root        : ${REPO_ROOT}"
log "python           : ${PYTHON_BIN}"
log "config           : ${CONFIG}"
log "per-task config  : ${PER_TASK_CONFIG}"
log "output dir       : ${OUT_DIR}"
log "source mean      : ${SOURCE_MEAN_LIST}"
log "target mean      : ${TARGET_MEAN_LIST}"
log "gamma            : ${GAMMA}"
log "omega            : ${OMEGA}"
log "J                : ${J}"
log "iterations       : ${ITERATIONS}"
log "batch size       : ${BATCH_SIZE}"
log "lr               : ${LR}"
log "num timesteps    : ${NUM_TIMESTEPS}"
log "schedule         : ${SCHEDULE}"
log "norm             : ${NORM}"
log "reference N      : ${REFERENCE_SAMPLES}"
log "heatmap N        : ${HEATMAP_SAMPLES}"
log "seeds            : ${SEEDS}"
log "device           : ${DEVICE:-auto}"
log "plot             : ${PLOT}"
log "dry run          : ${DRY_RUN}"
log "smoke            : ${SMOKE}"
log "======================================================================"

# -----------------------------------------------------------------------------
# 1) Run the toy experiment: source pretraining -> classifier -> Figure 2(a)
#    -> ANT transfer -> heat-map data. Writes toy_results.json, model
#    checkpoints and, when matplotlib is available, the Figure 2 panels.
# -----------------------------------------------------------------------------
EXIT_CODE=0
for SEED in ${SEEDS}; do
  SEED_OUT="${OUT_DIR}"
  if [[ "$(echo ${SEEDS} | wc -w)" -gt 1 ]]; then
    SEED_OUT="${OUT_DIR}/seed_${SEED}"
  fi
  mkdir -p "${SEED_OUT}" || true

  log "----------------------------------------------------------------------"
  log "Toy experiment seed=${SEED} -> ${SEED_OUT}"
  log "----------------------------------------------------------------------"

  TOY_ARGS=(
    --out-dir "${SEED_OUT}"
    --seed "${SEED}"
    --gamma "${GAMMA}"
    --omega "${OMEGA}"
    --J "${J}"
    --iterations "${ITERATIONS}"
    --batch-size "${BATCH_SIZE}"
    --lr "${LR}"
    --num-timesteps "${NUM_TIMESTEPS}"
    --schedule "${SCHEDULE}"
    --norm "${NORM}"
    --num-samples-reference "${REFERENCE_SAMPLES}"
    --heatmap-samples "${HEATMAP_SAMPLES}"
    --source-mean ${SOURCE_MEAN_LIST}
    --target-mean ${TARGET_MEAN_LIST}
    --config "${CONFIG}"
  )
  if [[ "${VERBOSE}" == "1" ]]; then
    TOY_ARGS+=(--verbose)
  fi
  if [[ "${PLOT}" != "1" ]]; then
    TOY_ARGS+=(--no-plots)
  fi
  if [[ -n "${DEVICE}" ]]; then
    TOY_ARGS+=(--device "${DEVICE}")
  fi

  if ! run "toy experiment (seed ${SEED})" -- "${PYTHON_BIN}" -m dpm_ant.toy.toy_2d "${TOY_ARGS[@]}"; then
    # Fall back to direct script execution (works even if the package is not
    # installed/importable in the current environment).
    if ! run "toy experiment via file (seed ${SEED})" -- "${PYTHON_BIN}" dpm_ant/toy/toy_2d.py "${TOY_ARGS[@]}"; then
      warn "toy experiment failed for seed ${SEED}"
      EXIT_CODE=1
      continue
    fi
  fi

  # 2) Re-render Figure 2 from the saved results if the automatic plotting was
  #    disabled (PLOT=0) or if a plot backend was unavailable during step 1.
  if [[ "${PLOT}" == "1" ]]; then
    RESULTS_JSON="${SEED_OUT}/toy_results.json"
    if [[ -f "${RESULTS_JSON}" ]]; then
      if ! run "render Figure 2 (seed ${SEED})" -- \
        "${PYTHON_BIN}" -m dpm_ant.toy.toy_plots \
        --results "${RESULTS_JSON}" \
        --out-dir "${SEED_OUT}" \
        --prefix figure2; then
        run "render Figure 2 via file (seed ${SEED})" -- \
          "${PYTHON_BIN}" dpm_ant/toy/toy_plots.py \
          --results "${RESULTS_JSON}" \
          --out-dir "${SEED_OUT}" \
          --prefix figure2 || warn "figure rendering failed for seed ${SEED}"
      fi
    else
      warn "results file not found: ${RESULTS_JSON} (skipping Figure 2 render)"
    fi
  fi
done

# -----------------------------------------------------------------------------
# 3) Summary
# -----------------------------------------------------------------------------
log "======================================================================"
if [[ "${EXIT_CODE}" -eq 0 ]]; then
  log "Toy experiment finished successfully."
else
  log "Toy experiment finished with errors (see log: ${LOG_FILE})."
fi
log "Artifacts under: ${OUT_DIR}"
log "  - toy_results.json            (all Figure 2 numeric payloads)"
log "  - toy_source_model.pt         (source-pretrained MLP)"
log "  - toy_ant_model.pt            (ANT-adapted MLP)"
log "  - figure2a.png / .pdf         (Figure 2(a): gradients + noise cloud)"
log "  - figure2b.png / .pdf         (Figure 2(b): baseline heat-map)"
log "  - figure2c.png / .pdf         (Figure 2(c): ANT heat-map)"
log "  - figure2_summary.json        (extracted numeric summary)"
log "  - logs/run_toy.log            (full stdout/stderr log)"
log ""
log "Expected qualitative outcome (Section 5.1):"
log "  * ANT gradient direction is closest to the 10k-sample reference (~SW)."
log "  * The adversarial-noise cloud is an ellipse whose principal axis"
log "    follows the model-parameter gradient (vs. a circle at init)."
log "  * The ANT heat-map shows a brighter central highlight and near-"
log "    parallel sampling trajectories compared to the baseline."
log "======================================================================"

exit "${EXIT_CODE}"
