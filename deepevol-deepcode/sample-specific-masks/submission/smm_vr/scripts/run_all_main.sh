#!/usr/bin/env bash
# =============================================================================
# run_all_main.sh -- Main-results driver for the SMM (ICML 2024) reproduction.
#
# Reproduces:
#   * Table 1 : SMM (ours) vs. Pad / Narrow / Medium / Full shared-mask VR
#               baselines with a frozen ImageNet-1K ResNet-18 and ResNet-50
#               over the 11 target datasets listed in Table 6.
#   * Table 2 : the same comparison with a frozen ImageNet-1K ViT-B/32
#               (input resolution 384x384, 6-layer mask generator).
#
# Optionally chains the auxiliary studies that accompany the main tables:
#   * Table 10 (Appendix D.1) -- label-mapping study (Rlm / Flm / Ilm).
#   * Table 11 (Appendix D.3) -- mask-generator scaling study (EuroSAT).
#   * Tables 13/14 (Appendix E) -- LoRA / Finetuning-FC comparisons.
#   * Table 12 (Appendix D.4) -- StanfordCars failure case.
#
# Usage:
#   bash smm_vr/scripts/run_all_main.sh [options]
#
# Options:
#   --mode MODE             all|table1|table2|label_mappings|scaling|finetuning|stanfordcars
#                           (default: all)
#   --backbones "A B ..."   resnet18 resnet50 vit_b32 (default: all three)
#   --datasets "A B ..."    11 target datasets (default: full Table 6 list)
#   --methods "A B ..."     ours pad narrow medium full (default: all five)
#   --seeds "0 1 2"         three seeds used for mean +/- std reporting
#   --label-mapping NAME    ilm (default) | flm | rlm
#   --patch-size N          SMM patch size (default: 8, i.e. l = 3)
#   --data-root DIR         dataset root
#   --output-dir DIR        results/log directory
#   --num-workers N         DataLoader workers (default: 4)
#   --device DEV            cuda|cpu|cuda:0 (default: auto)
#   --train-fraction F      deterministic class-balanced subsample (debug)
#   --no-extra              skip Table 10/11/12/13/14 auxiliary studies
#   -h|--help               this message
#
# Environment variables mirror every flag above:
#   MODE BACKBONES DATASETS METHODS SEEDS LABEL_MAPPING PATCH_SIZE DATA_ROOT
#   OUTPUT_DIR NUM_WORKERS DEVICE TRAIN_FRACTION MAX_TRAIN_BATCHES
#   MAX_EVAL_BATCHES NO_EXTRA EXTRA_ARGS PYTHON
#
# Everything is logged to <OUTPUT_DIR>/logs/run_all_main_<timestamp>.log.
# =============================================================================

set -o errexit
set -o nounset
set -o pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
MODE="${MODE:-all}"

BACKBONES="${BACKBONES:-resnet18 resnet50 vit_b32}"
METHODS="${METHODS:-ours pad narrow medium full}"
DATASETS="${DATASETS:-cifar10 cifar100 svhn gtsrb flowers102 dtd ucf101 food101 sun397 eurosat oxfordpets}"
SEEDS="${SEEDS:-0 1 2}"

LABEL_MAPPING="${LABEL_MAPPING:-ilm}"
PATCH_SIZE="${PATCH_SIZE:-8}"

NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-auto}"
TRAIN_FRACTION="${TRAIN_FRACTION:-}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-}"
NO_EXTRA="${NO_EXTRA:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

PYTHON="${PYTHON:-python}"

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_PATH}/.." && pwd)"     # .../smm_vr
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"        # repo root that contains smm_vr/

DATA_ROOT="${DATA_ROOT:-${SMM_DATA_ROOT:-${REPO_DIR}/data}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
LOG_DIR="${OUTPUT_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_all_main_${TIMESTAMP}.log"

export SMM_DATA_ROOT="${DATA_ROOT}"
export SMM_SEED="${SMM_SEED:-0}"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
usage() {
    sed -n '2,60p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

log() {
    printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${LOG_FILE}"
}

# run_stage <stage-name> <python-module> [args...]
# Executes one Python stage, tees its output to the shared log, and aborts the
# whole script when the stage exits non-zero (pipefail keeps tee from masking
# the Python exit status).
run_stage() {
    local stage_name="$1"
    shift
    log "==> STAGE: ${stage_name}"
    log "    CMD: ${PYTHON} -m $*"
    set +e
    "${PYTHON}" -m "$@" 2>&1 | tee -a "${LOG_FILE}"
    local status="${PIPESTATUS[0]}"
    set -e
    if [[ "${status}" -ne 0 ]]; then
        log "STAGE FAILED (exit ${status}): ${stage_name}"
        exit "${status}"
    fi
    log "<== STAGE DONE: ${stage_name}"
}

# build common CLI arguments shared by every runner
common_args() {
    local args=("--seeds" ${SEEDS} "--device" "${DEVICE}" "--num-workers" "${NUM_WORKERS}")
    args+=("--data-root" "${DATA_ROOT}")
    args+=("--output-dir" "${OUTPUT_DIR}")
    if [[ -n "${TRAIN_FRACTION}" ]]; then
        args+=("--train-fraction" "${TRAIN_FRACTION}")
    fi
    if [[ -n "${MAX_TRAIN_BATCHES}" ]]; then
        args+=("--max-train-batches" "${MAX_TRAIN_BATCHES}")
    fi
    if [[ -n "${MAX_EVAL_BATCHES}" ]]; then
        args+=("--max-eval-batches" "${MAX_EVAL_BATCHES}")
    fi
    printf '%s\n' "${args[@]}"
}

# --------------------------------------------------------------------------- #
# Argument parsing (CLI overrides environment variables)
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)           MODE="$2"; shift 2 ;;
        --backbones)      BACKBONES="$2"; shift 2 ;;
        --datasets)       DATASETS="$2"; shift 2 ;;
        --methods)        METHODS="$2"; shift 2 ;;
        --seeds)          SEEDS="$2"; shift 2 ;;
        --label-mapping)  LABEL_MAPPING="$2"; shift 2 ;;
        --patch-size)     PATCH_SIZE="$2"; shift 2 ;;
        --data-root)      DATA_ROOT="$2"; export SMM_DATA_ROOT="$2"; shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2"; LOG_DIR="$OUTPUT_DIR/logs"
                          LOG_FILE="${LOG_DIR}/run_all_main_${TIMESTAMP}.log"; shift 2 ;;
        --num-workers)    NUM_WORKERS="$2"; shift 2 ;;
        --device)         DEVICE="$2"; shift 2 ;;
        --train-fraction) TRAIN_FRACTION="$2"; shift 2 ;;
        --no-extra)       NO_EXTRA=1; shift ;;
        -h|--help)        usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

# Normalise aliases so every spelling maps onto the runner's canonical keys
case "${MODE}" in
    all|full)         MODE="all" ;;
    table1|resnet|resnets|rn) MODE="table1" ;;
    table2|vit|vit_b32)       MODE="table2" ;;
    table10|labels|label|label_mappings|label-mappings) MODE="label_mappings" ;;
    table11|scale|scaling)    MODE="scaling" ;;
    table13|table14|finetune|finetuning|lora) MODE="finetuning" ;;
    table12|cars|stanfordcars|failure) MODE="stanfordcars" ;;
    *) echo "Unknown --mode '${MODE}'" >&2; usage ;;
esac

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# --------------------------------------------------------------------------- #
# Banner
# --------------------------------------------------------------------------- #
{
    echo "=============================================================="
    echo " SMM (ICML 2024) -- main-results reproduction driver"
    echo "=============================================================="
    echo " mode            : ${MODE}"
    echo " backbones       : ${BACKBONES}"
    echo " datasets        : ${DATASETS}"
    echo " methods         : ${METHODS}"
    echo " seeds           : ${SEEDS}"
    echo " label mapping   : ${LABEL_MAPPING}"
    echo " patch size      : ${PATCH_SIZE}"
    echo " data root       : ${DATA_ROOT}"
    echo " output dir      : ${OUTPUT_DIR}"
    echo " device          : ${DEVICE}"
    echo " num workers     : ${NUM_WORKERS}"
    echo " extra studies   : NO_EXTRA=${NO_EXTRA}"
    echo " python          : ${PYTHON}"
    echo " log file        : ${LOG_FILE}"
    echo "=============================================================="
} | tee -a "${LOG_FILE}"

# --------------------------------------------------------------------------- #
# Stage 0 -- environment / protocol sanity checks
# --------------------------------------------------------------------------- #
log "==> STAGE: environment sanity check"

"${PYTHON}" - <<'PYEOF' 2>&1 | tee -a "${LOG_FILE}"
import sys

print("python:", sys.version.split()[0])
try:
    import torch
    print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as exc:  # pragma: no cover - diagnostics only
    print("torch import failed:", exc)

try:
    import torchvision
    print("torchvision:", torchvision.__version__)
except Exception as exc:  # pragma: no cover
    print("torchvision import failed:", exc)

# Table 4 parameter budgets for the mask generator f_mask
try:
    from smm_vr.models.mask_generator import (
        EXPECTED_PARAMETERS,
        build_mask_generator,
        count_parameters,
    )

    print("\nmask generator parameter budgets (paper Table 4):")
    ok = True
    for backbone, expected in EXPECTED_PARAMETERS.items():
        if backbone not in ("resnet18", "resnet50", "vit_b32"):
            continue
        model = build_mask_generator(backbone)
        got = count_parameters(model)
        flag = "OK" if got == expected else "MISMATCH"
        if got != expected:
            ok = False
        print(f"  {backbone:<10} expected={expected:<8} actual={got:<8} [{flag}]")
    if not ok:
        print("WARNING: mask generator parameter count differs from Table 4.")
except Exception as exc:  # pragma: no cover
    print("could not verify mask generator parameters:", exc)

# Reference averages (Tables 1 / 2) used to compare reproduced numbers
try:
    from smm_vr.engine.metrics import (
        TABLE1_RESNET18,
        TABLE1_RESNET50,
        TABLE2_VIT_B32,
    )

    def _avg(table):
        if isinstance(table, dict) and "average" in table:
            return table["average"]
        for key in ("average", "AVERAGE", "avg"):
            if isinstance(table, dict) and key in table:
                return table[key]
        return None

    print("\nreference averages:")
    print("  Table 1 ResNet-18 :", _avg(TABLE1_RESNET18))
    print("  Table 1 ResNet-50 :", _avg(TABLE1_RESNET50))
    print("  Table 2 ViT-B/32  :", _avg(TABLE2_VIT_B32))
except Exception as exc:  # pragma: no cover
    print("could not load reference tables:", exc)
PYEOF

# --------------------------------------------------------------------------- #
# Stage 1 -- Table 1 (ResNet-18 / ResNet-50) and Table 2 (ViT-B/32)
# --------------------------------------------------------------------------- #
run_main_for_backbone() {
    local backbone="$1"
    local stage_args=()
    mapfile -t stage_args < <(common_args)
    run_stage "main tables -- ${backbone}" \
        smm_vr.experiments.run_main \
        --backbone "${backbone}" \
        --datasets ${DATASETS} \
        --methods ${METHODS} \
        --label-mapping "${LABEL_MAPPING}" \
        --patch-size "${PATCH_SIZE}" \
        "${stage_args[@]}"
}

if [[ "${MODE}" == "all" || "${MODE}" == "table1" ]]; then
    for backbone in ${BACKBONES}; do
        case "${backbone}" in
            resnet18|resnet50)
                run_main_for_backbone "${backbone}" ;;
            vit_b32)
                [[ "${MODE}" == "table1" ]] && continue
                run_main_for_backbone "${backbone}" ;;
            *)
                log "WARNING: unsupported backbone '${backbone}' -- skipped" ;;
        esac
    done
fi

if [[ "${MODE}" == "table2" ]]; then
    run_main_for_backbone "vit_b32"
fi

# --------------------------------------------------------------------------- #
# Stage 2 -- optional auxiliary studies (Tables 10/11/12/13/14)
# --------------------------------------------------------------------------- #
if [[ "${NO_EXTRA}" -eq 0 && "${MODE}" == "all" ]]; then
    mapfile -t AUX_ARGS < <(common_args)

    # Table 10 (Appendix D.1): label-mapping study
    run_stage "Table 10 -- label-mapping study (Rlm / Flm / Ilm)" \
        smm_vr.experiments.run_label_mappings \
        "${AUX_ARGS[@]}"

    # Table 11 (Appendix D.3): mask-generator scaling on EuroSAT + ResNet-18
    run_stage "Table 11 -- masking-model scaling study (EuroSAT, ResNet-18)" \
        smm_vr.experiments.run_scaling \
        "${AUX_ARGS[@]}"

    # Tables 13/14 (Appendix E): LoRA (ViT-L) and Finetuning-FC comparisons
    run_stage "Tables 13/14 -- finetuning / LoRA comparisons" \
        smm_vr.experiments.run_finetuning \
        "${AUX_ARGS[@]}"

    # Table 12 (Appendix D.4): StanfordCars failure case
    run_stage "Table 12 -- StanfordCars failure case" \
        smm_vr.experiments.run_stanfordcars \
        "${AUX_ARGS[@]}"
fi

if [[ "${MODE}" == "label_mappings" ]]; then
    mapfile -t AUX_ARGS < <(common_args)
    run_stage "Table 10 -- label-mapping study (Rlm / Flm / Ilm)" \
        smm_vr.experiments.run_label_mappings "${AUX_ARGS[@]}"
fi

if [[ "${MODE}" == "scaling" ]]; then
    mapfile -t AUX_ARGS < <(common_args)
    run_stage "Table 11 -- masking-model scaling study (EuroSAT, ResNet-18)" \
        smm_vr.experiments.run_scaling "${AUX_ARGS[@]}"
fi

if [[ "${MODE}" == "finetuning" ]]; then
    mapfile -t AUX_ARGS < <(common_args)
    run_stage "Tables 13/14 -- finetuning / LoRA comparisons" \
        smm_vr.experiments.run_finetuning "${AUX_ARGS[@]}"
fi

if [[ "${MODE}" == "stanfordcars" ]]; then
    mapfile -t AUX_ARGS < <(common_args)
    run_stage "Table 12 -- StanfordCars failure case" \
        smm_vr.experiments.run_stanfordcars "${AUX_ARGS[@]}"
fi

# --------------------------------------------------------------------------- #
# Stage 3 -- summary of produced artifacts
# --------------------------------------------------------------------------- #
log "==> STAGE: summarising artifacts under ${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -maxdepth 2 -type f \
    \( -name '*.json' -o -name '*.txt' -o -name '*.pdf' -o -name '*.png' \) \
    -printf '  %p\n' 2>/dev/null | sort | tee -a "${LOG_FILE}" || true

log "All requested main-table stages completed."
log "Log: ${LOG_FILE}"
exit 0
