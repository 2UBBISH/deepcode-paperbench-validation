#!/usr/bin/env bash
#
# run_all_ablations.sh -- SMM (ICML 2024) ablation driver.
#
# Reproduces:
#   * Table 3  "Impact of Masking"   -- masking variants with frozen ResNet-18
#                 variants: only_delta | only_fmask | single_channel_fmask | ours
#                 expected averages: ours 52.53 > f_mask^s 49.70 > delta 46.85 > f_mask 42.59
#   * Figure 4 "Impact of Patch Size" -- l in {0,1,2,3,4}  => patch sizes {1,2,4,8,16}
#                 ResNet-18, default patch size 8 for all datasets.
#
# Both experiments use the Algorithm-1 protocol:
#   200 epochs, milestones (100, 145), batch 256 (64 for DTD / OxfordPets),
#   alpha_delta = 0.01 / gamma_delta = 0.1,
#   alpha_mask  = 0.01 / gamma_mask  = 0.1  (5-layer f_mask for ResNets),
#   label mapping = Ilm (refreshed every epoch), 3 seeds {0,1,2}.
#
# Usage:
#   bash smm_vr/scripts/run_all_ablations.sh                 # everything
#   bash smm_vr/scripts/run_all_ablations.sh --mode masking  # Table 3 only
#   bash smm_vr/scripts/run_all_ablations.sh --mode patch    # Figure 4 only
#   MODE=patch SEEDS="0 1 2" DATASETS="cifar10 eurosat" \
#       bash smm_vr/scripts/run_all_ablations.sh
#
# Environment variables (all optional):
#   MODE            all | masking | patch            (default: all)
#   BACKBONE        resnet18 (ablation default)      (default: resnet18)
#   DATASETS        space separated masking datasets (default: every main dataset)
#   PATCH_DATASETS  space separated patch-sweep datasets
#   L_VALUES        space separated l values         (default: 0 1 2 3 4)
#   SEEDS           space separated seeds            (default: 0 1 2)
#   DATA_ROOT       dataset download/root directory  (default: ./data)
#   OUTPUT_DIR      results directory                (default: ./outputs)
#   NUM_WORKERS     dataloader workers               (default: 4)
#   DEVICE          cuda / cpu / auto                (default: auto)
#   TRAIN_FRACTION  optional debug subsampling fraction (e.g. 0.05)
#   MAX_TRAIN_BATCHES / MAX_EVAL_BATCHES  debug batch limits
#   PYTHON          python interpreter               (default: python)
#   NO_PLOT         1 to skip the Figure-4 plotting step
#   EXTRA_ARGS      extra args forwarded to the runner
#
# Exit status: 0 on success, non-zero if any stage fails.

set -o errexit
set -o nounset
set -o pipefail

# --------------------------------------------------------------------------- #
# Locate the project root (directory that contains the `smm_vr` package).
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd -P)"
cd -- "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python}"

# --------------------------------------------------------------------------- #
# Defaults / environment overrides
# --------------------------------------------------------------------------- #
MODE="${MODE:-all}"
BACKBONE="${BACKBONE:-resnet18}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-0 1 2}"
L_VALUES="${L_VALUES:-0 1 2 3 4}"
PATCH_DATASETS="${PATCH_DATASETS:-cifar10 svhn flowers102 eurosat}"
DATASETS="${DATASETS:-cifar10 cifar100 svhn gtsrb flowers102 dtd ucf101 food101 sun397 eurosat oxfordpets}"
TRAIN_FRACTION="${TRAIN_FRACTION:-}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-}"
NO_PLOT="${NO_PLOT:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Ablation variant order taken from the paper (Table 3 ordering).
VARIANTS="${VARIANTS:-only_delta only_fmask single_channel_fmask ours}"

# --------------------------------------------------------------------------- #
# Tiny CLI (overrides env vars)
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)            MODE="$2"; shift 2 ;;
    --backbone)        BACKBONE="$2"; shift 2 ;;
    --datasets)        DATASETS="$2"; shift 2 ;;
    --patch-datasets)  PATCH_DATASETS="$2"; shift 2 ;;
    --l-values)        L_VALUES="$2"; shift 2 ;;
    --seeds)           SEEDS="$2"; shift 2 ;;
    --data-root)       DATA_ROOT="$2"; shift 2 ;;
    --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
    --num-workers)     NUM_WORKERS="$2"; shift 2 ;;
    --device)          DEVICE="$2"; shift 2 ;;
    --train-fraction)  TRAIN_FRACTION="$2"; shift 2 ;;
    --no-plot)         NO_PLOT=1; shift ;;
    -h|--help)
      sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "[warn] unknown argument: $1" >&2; shift ;;
  esac
done

case "${MODE}" in
  all|masking|patch|patch_size|table3|figure4) ;;
  *) echo "[error] MODE must be one of: all | masking | patch | table3 | figure4" >&2; exit 2 ;;
esac

# Normalise the mode onto the runner's vocabulary.
case "${MODE}" in
  table3)      RUNNER_MODE="masking" ;;
  figure4)     RUNNER_MODE="patch_size" ;;
  patch)       RUNNER_MODE="patch_size" ;;
  *)           RUNNER_MODE="${MODE}" ;;
esac

mkdir -p -- "${OUTPUT_DIR}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p -- "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

echo "======================================================================"
echo " SMM ablation experiments"
echo "----------------------------------------------------------------------"
echo " project root   : ${PROJECT_ROOT}"
echo " python         : ${PYTHON}"
echo " mode           : ${MODE} (runner mode: ${RUNNER_MODE})"
echo " backbone       : ${BACKBONE}"
echo " seeds          : ${SEEDS}"
echo " masking dsets  : ${DATASETS}"
echo " patch dsets    : ${PATCH_DATASETS}"
echo " l values       : ${L_VALUES}"
echo " data root      : ${DATA_ROOT}"
echo " output dir     : ${OUTPUT_DIR}"
echo " device         : ${DEVICE}"
echo "======================================================================"

# --------------------------------------------------------------------------- #
# Sanity check: package importable + mask-generator parameter budgets.
# --------------------------------------------------------------------------- #
echo "[1/3] environment / parameter check"
"${PYTHON}" - <<'PY'
import sys
try:
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover - environment guard
    print(f"[error] torch is not importable: {exc}", file=sys.stderr)
    sys.exit(1)

from smm_vr.models.mask_generator import EXPECTED_PARAMETERS, build_mask_generator, count_parameters

ok = True
for backbone, expected in EXPECTED_PARAMETERS.items():
    try:
        gen = build_mask_generator(backbone)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not build mask generator for {backbone}: {exc}")
        continue
    got = count_parameters(gen)
    status = "ok" if got == expected else "MISMATCH"
    if got != expected:
        ok = False
    print(f"    f_mask[{backbone:<11}] params={got:>7d} expected={expected:>7d} -> {status}")
if not ok:
    print("[warn] mask-generator parameter budgets do not match Table 4", file=sys.stderr)
PY

# --------------------------------------------------------------------------- #
# Build the shared argument list.
# --------------------------------------------------------------------------- #
COMMON_ARGS=(
  --backbone "${BACKBONE}"
  --seeds ${SEEDS}
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
)
[[ -n "${TRAIN_FRACTION}" ]]  && COMMON_ARGS+=(--train-fraction "${TRAIN_FRACTION}")
[[ -n "${MAX_TRAIN_BATCHES}" ]] && COMMON_ARGS+=(--max-train-batches "${MAX_TRAIN_BATCHES}")
[[ -n "${MAX_EVAL_BATCHES}" ]]  && COMMON_ARGS+=(--max-eval-batches "${MAX_EVAL_BATCHES}")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS}" ]] && COMMON_ARGS+=(${EXTRA_ARGS})

run_stage () {
  local stage_name="$1"; shift
  local log_file="${LOG_DIR}/${stage_name}_${TIMESTAMP}.log"
  echo "[run] ${stage_name}  (log: ${log_file})"
  # shellcheck disable=SC2068
  "${PYTHON}" -m smm_vr.experiments.run_ablations "$@" 2>&1 | tee "${log_file}"
}

# --------------------------------------------------------------------------- #
# Stage 2: ablations
# --------------------------------------------------------------------------- #
echo "[2/3] running ${RUNNER_MODE} ablation(s)"
if [[ "${RUNNER_MODE}" == "all" || "${RUNNER_MODE}" == "masking" ]]; then
  run_stage "masking" \
    --mode masking \
    --datasets ${DATASETS} \
    --variants ${VARIANTS} \
    "${COMMON_ARGS[@]}"
fi

if [[ "${RUNNER_MODE}" == "all" || "${RUNNER_MODE}" == "patch_size" ]]; then
  run_stage "patch_size" \
    --mode patch_size \
    --patch-datasets ${PATCH_DATASETS} \
    --l-values ${L_VALUES} \
    "${COMMON_ARGS[@]}"
fi

# --------------------------------------------------------------------------- #
# Stage 3: Figure 4 curve plot (optional)
# --------------------------------------------------------------------------- #
if [[ "${RUNNER_MODE}" == "all" || "${RUNNER_MODE}" == "patch_size" ]]; then
  if [[ "${NO_PLOT}" != "1" ]]; then
    echo "[3/3] plotting patch-size curves (Figure 4)"
    CURVES="${OUTPUT_DIR}/patch_size_curves.json"
    FIG_DIR="${OUTPUT_DIR}/figures"
    mkdir -p -- "${FIG_DIR}"
    if [[ -f "${CURVES}" ]]; then
      if ! "${PYTHON}" -m smm_vr.analysis.plot_patch_size \
              --curves "${CURVES}" \
              --output "${FIG_DIR}/figure4_patch_size.pdf" 2>&1 | tee "${LOG_DIR}/plot_patch_size_${TIMESTAMP}.log"; then
        echo "[warn] plot_patch_size failed; curves JSON is still available at ${CURVES}" >&2
      fi
    else
      echo "[warn] ${CURVES} not found; skipping the Figure-4 plot" >&2
    fi
  else
    echo "[3/3] plotting skipped (NO_PLOT=1)"
  fi
else
  echo "[3/3] plotting skipped (masking-only run)"
fi

echo "======================================================================"
echo " ablations finished"
echo "   results : ${OUTPUT_DIR}"
echo "   logs    : ${LOG_DIR}"
echo "======================================================================"
